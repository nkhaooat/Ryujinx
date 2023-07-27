using Ryujinx.Common.Logging;
using Ryujinx.Graphics.GAL;
using Ryujinx.Graphics.Gpu.Engine.Types;
using Ryujinx.Graphics.Gpu.Image;
using Ryujinx.Graphics.Gpu.Shader;
using Ryujinx.Graphics.Shader;
using Ryujinx.Graphics.Shader.Translation;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace Ryujinx.Graphics.Gpu.Engine.Threed
{
    class VtgAsCompute
    {
        private const int ComputeLocalSize = 32;

        private const int DummyBufferSize = 16;

        private readonly GpuContext _context;
        private readonly GpuChannel _channel;
        private readonly DeviceStateWithShadow<ThreedClassState> _state;

        private class BufferTextureCache
        {
            private readonly Dictionary<Format, ITexture> _cache;

            public BufferTextureCache()
            {
                _cache = new();
            }

            public ITexture Get(IRenderer renderer, Format format)
            {
                if (!_cache.TryGetValue(format, out ITexture bufferTexture))
                {
                    bufferTexture = renderer.CreateTexture(new TextureCreateInfo(
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        format,
                        DepthStencilMode.Depth,
                        Target.TextureBuffer,
                        SwizzleComponent.Red,
                        SwizzleComponent.Green,
                        SwizzleComponent.Blue,
                        SwizzleComponent.Alpha));

                    _cache.Add(format, bufferTexture);
                }

                return bufferTexture;
            }
        }

        private struct Buffer
        {
            public BufferHandle Handle;
            public int Size;
        }

        private struct IndexBuffer
        {
            public BufferHandle Handle { get; }
            public int Count { get; }
            public int Size { get; }

            public IndexBuffer(BufferHandle handle, int count, int size)
            {
                Handle = handle;
                Count = count;
                Size = size;
            }

            public BufferRange ToRange()
            {
                return new BufferRange(Handle, 0, Size);
            }

            public BufferRange ToRange(int size)
            {
                return new BufferRange(Handle, 0, size);
            }
        }

        private BufferTextureCache[] _bufferTextures;
        private BufferHandle _dummyBuffer;
        private BufferHandle _vertexInfoBuffer;
        private Buffer _vertexDataBuffer;
        private Buffer _geometryVertexDataBuffer;
        private Buffer _geometryIndexDataBuffer;
        private BufferHandle _sequentialIndexBuffer;
        private int _sequentialIndexBufferCount;

        private readonly Dictionary<PrimitiveTopology, IndexBuffer> _topologyRemapBuffers;

        public VtgAsCompute(GpuContext context, GpuChannel channel, DeviceStateWithShadow<ThreedClassState> state)
        {
            _context = context;
            _channel = channel;
            _state = state;
            _bufferTextures = new BufferTextureCache[Constants.TotalVertexBuffers + 2];
            _topologyRemapBuffers = new();
        }

        public void DrawAsCompute(
            ShaderAsCompute vertexAsCompute,
            ShaderAsCompute geometryAsCompute,
            IProgram vertexPassthroughProgram,
            PrimitiveTopology topology,
            int count,
            int instanceCount,
            int firstIndex,
            int firstVertex,
            int firstInstance,
            bool indexed)
        {
            _context.Renderer.Pipeline.SetProgram(vertexAsCompute.HostProgram);

            Span<int> vertexInfo = stackalloc int[8 + 32 * 4 * 2];

            int primitivesCount = GetPrimitivesCount(topology, count);

            vertexInfo[0] = count;
            vertexInfo[1] = instanceCount;
            vertexInfo[2] = firstVertex;
            vertexInfo[3] = firstInstance;
            vertexInfo[4] = primitivesCount;

            for (int index = 0; index < Constants.TotalVertexAttribs; index++)
            {
                var vertexAttrib = _state.State.VertexAttribState[index];

                if (!FormatTable.TryGetSingleComponentAttribFormat(vertexAttrib.UnpackFormat(), out Format format, out int componentsCount))
                {
                    Logger.Debug?.Print(LogClass.Gpu, $"Invalid attribute format 0x{vertexAttrib.UnpackFormat():X}.");

                    format = vertexAttrib.UnpackType() switch
                    {
                        VertexAttribType.Sint => Format.R32Sint,
                        VertexAttribType.Uint => Format.R32Uint,
                        _ => Format.R32Float
                    };

                    componentsCount = 4;
                }

                for (int c = 1; c < componentsCount; c++)
                {
                    vertexInfo[8 + index * 4 + c] = 1;
                }

                if (vertexAttrib.UnpackIsConstant())
                {
                    SetDummyBufferTexture(vertexAsCompute.Reservations, index, format);
                    continue;
                }

                int bufferIndex = vertexAttrib.UnpackBufferIndex();

                GpuVa endAddress = _state.State.VertexBufferEndAddress[bufferIndex];
                var vertexBuffer = _state.State.VertexBufferState[bufferIndex];
                bool instanced = _state.State.VertexBufferInstanced[bufferIndex];

                ulong address = vertexBuffer.Address.Pack();

                if (!vertexBuffer.UnpackEnable() || !_channel.MemoryManager.IsMapped(address))
                {
                    SetDummyBufferTexture(vertexAsCompute.Reservations, index, format);
                    continue;
                }

                int vbStride = vertexBuffer.UnpackStride();
                ulong vbSize = GetVertexBufferSize(address, endAddress.Pack(), vbStride, indexed, instanced, firstVertex, count);

                ulong oldVbSize = vbSize;

                ulong attributeOffset = (ulong)firstVertex * (ulong)vbStride + (ulong)vertexAttrib.UnpackOffset();
                int componentSize = format.GetScalarSize();

                address += attributeOffset;
                vbSize = Align(vbSize - attributeOffset, componentSize);

                SetBufferTexture(vertexAsCompute.Reservations, index, format, address, vbSize);

                vertexInfo[8 + index * 4] = vbStride / componentSize;
                vertexInfo[8 + 32 * 4 + index * 4] = instanced ? vertexBuffer.Divisor : 0;
            }

            if (indexed)
            {
                SetIndexBufferTexture(vertexAsCompute.Reservations, firstIndex, count);
            }
            else
            {
                SetSequentialIndexBufferTexture(vertexAsCompute.Reservations, count);
            }

            int vertexInfoBinding = vertexAsCompute.Reservations.GetVertexInfoConstantBufferBinding();
            BufferRange vertexInfoRange = new(PushVertexInfo(vertexInfo), 0, vertexInfo.Length * sizeof(int));
            _context.Renderer.Pipeline.SetUniformBuffers(stackalloc[] { new BufferAssignment(vertexInfoBinding, vertexInfoRange) });

            int vertexDataBinding = vertexAsCompute.Reservations.GetVertexOutputStorageBufferBinding();
            int vertexDataSize = vertexAsCompute.Reservations.OutputSizeInBytesPerInvocation * count * instanceCount;
            _context.Renderer.Pipeline.SetStorageBuffers(stackalloc[] { new BufferAssignment(vertexDataBinding, EnsureBuffer(ref _vertexDataBuffer, vertexDataSize)) });

            _context.Renderer.Pipeline.DispatchCompute(
                (count + ComputeLocalSize - 1) / ComputeLocalSize,
                (instanceCount + ComputeLocalSize - 1) / ComputeLocalSize,
                1);

            // Wait until compute is done.
            // TODO: Batch compute and draw operations to avoid pipeline stalls.
            _context.Renderer.Pipeline.Barrier();

            if (geometryAsCompute != null)
            {
                int totalPrimitivesCount = GetPrimitivesCount(topology, count * instanceCount);
                int maxCompleteStrips = GetMaxCompleteStrips(geometryAsCompute.Info.GeometryVerticesPerPrimitive, geometryAsCompute.Info.GeometryMaxOutputVertices);
                int totalVerticesCount = totalPrimitivesCount * geometryAsCompute.Info.GeometryMaxOutputVertices;
                int geometryVbDataSize = totalVerticesCount * geometryAsCompute.Reservations.OutputSizeInBytesPerInvocation;
                int geometryIbDataCount = totalVerticesCount + totalPrimitivesCount * maxCompleteStrips;
                int geometryIbDataSize = geometryIbDataCount * sizeof(uint);

                _context.Renderer.Pipeline.SetProgram(geometryAsCompute.HostProgram);

                SetTopologyRemapBufferTexture(geometryAsCompute.Reservations, topology, count);

                int geometryVbBinding = geometryAsCompute.Reservations.GetGeometryVertexOutputStorageBufferBinding();
                int geometryIbBinding = geometryAsCompute.Reservations.GetGeometryIndexOutputStorageBufferBinding();

                BufferRange vertexBuffer = EnsureBuffer(ref _geometryVertexDataBuffer, geometryVbDataSize);
                BufferRange indexBuffer = EnsureBuffer(ref _geometryIndexDataBuffer, geometryIbDataSize);

                _context.Renderer.Pipeline.SetStorageBuffers(stackalloc[]
                {
                    new BufferAssignment(geometryVbBinding, vertexBuffer),
                    new BufferAssignment(geometryIbBinding, indexBuffer),
                });

                _context.Renderer.Pipeline.DispatchCompute(
                    (primitivesCount + ComputeLocalSize - 1) / ComputeLocalSize,
                    (instanceCount + ComputeLocalSize - 1) / ComputeLocalSize,
                    1);

                _context.Renderer.Pipeline.Barrier();

                _context.Renderer.Pipeline.SetProgram(vertexPassthroughProgram);
                _context.Renderer.Pipeline.SetIndexBuffer(indexBuffer, IndexType.UInt);
                _context.Renderer.Pipeline.SetStorageBuffers(stackalloc[] { new BufferAssignment(vertexDataBinding, vertexBuffer) });

                // For instanced draws, we need to ensure that strip topologies will
                // restart on each new instance.
                _context.Renderer.Pipeline.SetPrimitiveRestart(true, -1);
                _context.Renderer.Pipeline.SetPrimitiveTopology(geometryAsCompute.Info.GeometryVerticesPerPrimitive switch
                {
                    3 => PrimitiveTopology.TriangleStrip,
                    2 => PrimitiveTopology.LineStrip,
                    _ => PrimitiveTopology.Points,
                });

                _context.Renderer.Pipeline.DrawIndexed(geometryIbDataCount, 1, 0, 0, 0);

                // TODO: Properly restore state.
                _context.Renderer.Pipeline.SetPrimitiveRestart(false, 0);
            }
            else
            {
                _context.Renderer.Pipeline.SetProgram(vertexPassthroughProgram);
                _context.Renderer.Pipeline.SetPrimitiveTopology(topology);
                _context.Renderer.Pipeline.Draw(count, instanceCount, 0, 0);
            }
        }

        private static int GetPrimitivesCount(PrimitiveTopology primitiveType, int count)
        {
            return primitiveType switch
            {
                PrimitiveTopology.Lines => count / 2,
                PrimitiveTopology.LinesAdjacency => count / 4,
                PrimitiveTopology.LineLoop => count > 1 ? count : 0,
                PrimitiveTopology.LineStrip => Math.Max(count - 1, 0),
                PrimitiveTopology.LineStripAdjacency => Math.Max(count - 3, 0),
                PrimitiveTopology.Triangles => count / 3,
                PrimitiveTopology.TrianglesAdjacency => count / 6,
                PrimitiveTopology.TriangleStrip or
                PrimitiveTopology.TriangleFan or
                PrimitiveTopology.Polygon => Math.Max(count - 2, 0),
                PrimitiveTopology.TriangleStripAdjacency => Math.Max(count - 2, 0) / 2,
                PrimitiveTopology.Quads => (count / 4) * 2, // In triangles.
                PrimitiveTopology.QuadStrip => Math.Max((count - 2) / 2, 0) * 2, // In triangles.
                _ => count,
            };
        }

        private static int GetVerticesPerPrimitive(PrimitiveTopology primitiveType)
        {
            return primitiveType switch
            {
                PrimitiveTopology.Lines or
                PrimitiveTopology.LineLoop or
                PrimitiveTopology.LineStrip => 2,
                PrimitiveTopology.LinesAdjacency or
                PrimitiveTopology.LineStripAdjacency => 4,
                PrimitiveTopology.Triangles or
                PrimitiveTopology.TriangleStrip or
                PrimitiveTopology.TriangleFan or
                PrimitiveTopology.Polygon => 3,
                PrimitiveTopology.TrianglesAdjacency or
                PrimitiveTopology.TriangleStripAdjacency => 6,
                PrimitiveTopology.Quads or
                PrimitiveTopology.QuadStrip => 3, // 2 triangles.
                _ => 1,
            };
        }

        private static int GetMaxCompleteStrips(int verticesPerPrimitive, int maxOutputVertices)
        {
            return maxOutputVertices / verticesPerPrimitive;
        }

        private void SetDummyBufferTexture(ResourceReservations reservations, int index, Format format)
        {
            if (_dummyBuffer == BufferHandle.Null)
            {
                _dummyBuffer = _context.Renderer.CreateBuffer(DummyBufferSize);
                _context.Renderer.Pipeline.ClearBuffer(_dummyBuffer, 0, DummyBufferSize, 0);
            }

            ITexture bufferTexture = EnsureBufferTexutre(index + 2, format);
            bufferTexture.SetStorage(new BufferRange(_dummyBuffer, 0, DummyBufferSize));

            _context.Renderer.Pipeline.SetTextureAndSampler(ShaderStage.Compute, reservations.GetVertexBufferTextureBinding(index), bufferTexture, null);
        }

        private void SetBufferTexture(ResourceReservations reservations, int index, Format format, ulong address, ulong size)
        {
            var memoryManager = _channel.MemoryManager;

            address = memoryManager.Translate(address);
            BufferRange range = memoryManager.Physical.BufferCache.GetBufferRange(address, size);

            ITexture bufferTexture = EnsureBufferTexutre(index + 2, format);
            bufferTexture.SetStorage(range);

            _context.Renderer.Pipeline.SetTextureAndSampler(ShaderStage.Compute, reservations.GetVertexBufferTextureBinding(index), bufferTexture, null);
        }

        private void SetIndexBufferTexture(ResourceReservations reservations, int firstIndex, int count)
        {
            ulong address = _state.State.IndexBufferState.Address.Pack();
            ulong indexOffset = (ulong)firstIndex;
            ulong size = (ulong)count;

            Format format = Format.R8Uint;

            switch (_state.State.IndexBufferState.Type)
            {
                case IndexType.UShort:
                    indexOffset <<= 1;
                    size <<= 1;
                    format = Format.R16Uint;
                    break;
                case IndexType.UInt:
                    indexOffset <<= 2;
                    size <<= 2;
                    format = Format.R32Uint;
                    break;
            }

            var memoryManager = _channel.MemoryManager;

            address = memoryManager.Translate(address + indexOffset);
            BufferRange range = memoryManager.Physical.BufferCache.GetBufferRange(address, size);

            ITexture bufferTexture = EnsureBufferTexutre(0, format);
            bufferTexture.SetStorage(range);

            _context.Renderer.Pipeline.SetTextureAndSampler(ShaderStage.Compute, reservations.GetIndexBufferTextureBinding(), bufferTexture, null);
        }

        private void SetTopologyRemapBufferTexture(ResourceReservations reservations, PrimitiveTopology topology, int count)
        {
            ITexture bufferTexture = EnsureBufferTexutre(1, Format.R32Uint);
            bufferTexture.SetStorage(GetOrCreateTopologyRemapBuffer(topology, count));

            _context.Renderer.Pipeline.SetTextureAndSampler(ShaderStage.Compute, reservations.GetTopologyRemapBufferTextureBinding(), bufferTexture, null);
        }

        private void SetSequentialIndexBufferTexture(ResourceReservations reservations, int count)
        {
            ulong size = (ulong)count << 2;

            if (_sequentialIndexBufferCount < count)
            {
                if (_sequentialIndexBuffer != BufferHandle.Null)
                {
                    _context.Renderer.DeleteBuffer(_sequentialIndexBuffer);
                }

                _sequentialIndexBuffer = _context.Renderer.CreateBuffer(count * sizeof(uint));
                _sequentialIndexBufferCount = count;

                Span<int> data = new int[count];

                for (int index = 0; index < count; index++)
                {
                    data[index] = index;
                }

                _context.Renderer.SetBufferData(_sequentialIndexBuffer, 0, MemoryMarshal.Cast<int, byte>(data));
            }

            ITexture bufferTexture = EnsureBufferTexutre(0, Format.R32Uint);
            bufferTexture.SetStorage(new BufferRange(_sequentialIndexBuffer, 0, count * sizeof(uint)));

            _context.Renderer.Pipeline.SetTextureAndSampler(ShaderStage.Compute, reservations.GetIndexBufferTextureBinding(), bufferTexture, null);
        }

        private BufferRange GetOrCreateTopologyRemapBuffer(PrimitiveTopology topology, int count)
        {
            if (!_topologyRemapBuffers.TryGetValue(topology, out IndexBuffer buffer) || buffer.Count < count)
            {
                if (buffer.Handle != BufferHandle.Null)
                {
                    _context.Renderer.DeleteBuffer(buffer.Handle);
                }

                buffer = CreateTopologyRemapBuffer(topology, count);
                _topologyRemapBuffers[topology] = buffer;

                return buffer.ToRange();
            }

            return buffer.ToRange(Math.Max(GetPrimitivesCount(topology, count) * GetVerticesPerPrimitive(topology), 1) * sizeof(uint));
        }

        private IndexBuffer CreateTopologyRemapBuffer(PrimitiveTopology topology, int count)
        {
            // Size can't be zero as creating zero sized buffers is invalid.
            Span<int> data = new int[Math.Max(GetPrimitivesCount(topology, count) * GetVerticesPerPrimitive(topology), 1)];

            switch (topology)
            {
                case PrimitiveTopology.Points:
                case PrimitiveTopology.Lines:
                case PrimitiveTopology.LinesAdjacency:
                case PrimitiveTopology.Triangles:
                case PrimitiveTopology.TrianglesAdjacency:
                case PrimitiveTopology.Patches:
                    for (int index = 0; index < data.Length; index++)
                    {
                        data[index] = index;
                    }
                    break;
                case PrimitiveTopology.LineLoop:
                    data[data.Length - 1] = 0;

                    for (int index = 0; index < ((data.Length - 1) & ~1); index += 2)
                    {
                        data[index] = index >> 1;
                        data[index + 1] = (index >> 1) + 1;
                    }
                    break;
                case PrimitiveTopology.LineStrip:
                    for (int index = 0; index < ((data.Length - 1) & ~1); index += 2)
                    {
                        data[index] = index >> 1;
                        data[index + 1] = (index >> 1) + 1;
                    }
                    break;
                case PrimitiveTopology.TriangleStrip:
                    int tsTrianglesCount = data.Length / 3;
                    int tsOutIndex = 3;

                    if (tsTrianglesCount > 0)
                    {
                        data[0] = 0;
                        data[1] = 1;
                        data[2] = 2;
                    }

                    for (int tri = 1; tri < tsTrianglesCount; tri++)
                    {
                        int baseIndex = tri * 3;

                        if ((tri & 1) != 0)
                        {
                            data[baseIndex] = tsOutIndex - 1;
                            data[baseIndex + 1] = tsOutIndex - 2;
                            data[baseIndex + 2] = tsOutIndex++;
                        }
                        else
                        {
                            data[baseIndex] = tsOutIndex - 2;
                            data[baseIndex + 1] = tsOutIndex - 1;
                            data[baseIndex + 2] = tsOutIndex++;
                        }
                    }
                    break;
                case PrimitiveTopology.TriangleFan:
                case PrimitiveTopology.Polygon:
                    int tfTrianglesCount = data.Length / 3;
                    int tfOutIndex = 1;

                    for (int index = 0; index < tfTrianglesCount * 3; index += 3)
                    {
                        data[index] = 0;
                        data[index + 1] = tfOutIndex;
                        data[index + 2] = ++tfOutIndex;
                    }
                    break;
                case PrimitiveTopology.Quads:
                    int qQuadsCount = data.Length / 6;

                    for (int quad = 0; quad < qQuadsCount; quad++)
                    {
                        int index = quad * 6;
                        int qIndex = quad * 4;

                        data[index] = qIndex;
                        data[index + 1] = qIndex + 1;
                        data[index + 2] = qIndex + 2;
                        data[index + 3] = qIndex;
                        data[index + 4] = qIndex + 2;
                        data[index + 5] = qIndex + 3;
                    }
                    break;
                case PrimitiveTopology.QuadStrip:
                    int qsQuadsCount = data.Length / 6;

                    if (qsQuadsCount > 0)
                    {
                        data[0] = 0;
                        data[1] = 1;
                        data[2] = 2;
                        data[3] = 0;
                        data[4] = 2;
                        data[5] = 3;
                    }

                    for (int quad = 1; quad < qsQuadsCount; quad++)
                    {
                        int index = quad * 6;
                        int qIndex = quad * 2;

                        data[index] = qIndex + 1;
                        data[index + 1] = qIndex;
                        data[index + 2] = qIndex + 2;
                        data[index + 3] = qIndex + 1;
                        data[index + 4] = qIndex + 2;
                        data[index + 5] = qIndex + 3;
                    }
                    break;
                case PrimitiveTopology.LineStripAdjacency:
                    for (int index = 0; index < ((data.Length - 3) & ~3); index += 4)
                    {
                        int lIndex = index >> 2;

                        data[index] = lIndex;
                        data[index + 1] = lIndex + 1;
                        data[index + 2] = lIndex + 2;
                        data[index + 3] = lIndex + 3;
                    }
                    break;
                case PrimitiveTopology.TriangleStripAdjacency:
                    int tsaTrianglesCount = data.Length / 6;
                    int tsaOutIndex = 6;

                    if (tsaTrianglesCount > 0)
                    {
                        data[0] = 0;
                        data[1] = 1;
                        data[2] = 2;
                        data[3] = 3;
                        data[4] = 4;
                        data[5] = 5;
                    }

                    for (int tri = 1; tri < tsaTrianglesCount; tri++)
                    {
                        int baseIndex = tri * 6;

                        if ((tri & 1) != 0)
                        {
                            data[baseIndex] = tsaOutIndex - 2;
                            data[baseIndex + 1] = tsaOutIndex - 1;
                            data[baseIndex + 2] = tsaOutIndex - 4;
                            data[baseIndex + 3] = tsaOutIndex - 3;
                            data[baseIndex + 4] = tsaOutIndex++;
                            data[baseIndex + 5] = tsaOutIndex++;
                        }
                        else
                        {
                            data[baseIndex] = tsaOutIndex - 4;
                            data[baseIndex + 1] = tsaOutIndex - 3;
                            data[baseIndex + 2] = tsaOutIndex - 2;
                            data[baseIndex + 3] = tsaOutIndex - 1;
                            data[baseIndex + 4] = tsaOutIndex++;
                            data[baseIndex + 5] = tsaOutIndex++;
                        }
                    }
                    break;
            }

            ReadOnlySpan<byte> dataBytes = MemoryMarshal.Cast<int, byte>(data);

            BufferHandle buffer = _context.Renderer.CreateBuffer(dataBytes.Length);
            _context.Renderer.SetBufferData(buffer, 0, dataBytes);

            return new IndexBuffer(buffer, count, dataBytes.Length);
        }

        private ITexture EnsureBufferTexutre(int index, Format format)
        {
            return (_bufferTextures[index] ??= new()).Get(_context.Renderer, format);
        }

        private BufferHandle PushVertexInfo(ReadOnlySpan<int> data)
        {
            ReadOnlySpan<byte> dataAsByte = MemoryMarshal.Cast<int, byte>(data);

            if (_vertexInfoBuffer == BufferHandle.Null)
            {
                _vertexInfoBuffer = _context.Renderer.CreateBuffer(dataAsByte.Length);
            }

            _context.Renderer.SetBufferData(_vertexInfoBuffer, 0, dataAsByte);

            return _vertexInfoBuffer;
        }

        private BufferRange EnsureBuffer(ref Buffer buffer, int size)
        {
            if (buffer.Size < size)
            {
                if (buffer.Handle != BufferHandle.Null)
                {
                    _context.Renderer.DeleteBuffer(buffer.Handle);
                }

                buffer.Handle = _context.Renderer.CreateBuffer(size);
                buffer.Size = size;
            }

            return new BufferRange(buffer.Handle, 0, buffer.Size);
        }

        private ulong GetVertexBufferSize(ulong vbAddress, ulong vbEndAddress, int vbStride, bool indexed, bool instanced, int firstVertex, int vertexCount)
        {
            IndexType indexType = _state.State.IndexBufferState.Type;
            bool indexTypeSmall = indexType == IndexType.UByte || indexType == IndexType.UShort;
            ulong vbSize = vbEndAddress - vbAddress + 1;
            ulong size;

            if (indexed || vbStride == 0 || instanced)
            {
                // This size may be (much) larger than the real vertex buffer size.
                // Avoid calculating it this way, unless we don't have any other option.

                size = vbSize;

                if (vbStride > 0 && indexTypeSmall && indexed && !instanced)
                {
                    // If the index type is a small integer type, then we might be still able
                    // to reduce the vertex buffer size based on the maximum possible index value.

                    ulong maxVertexBufferSize = indexType == IndexType.UByte ? 0x100UL : 0x10000UL;

                    maxVertexBufferSize += _state.State.FirstVertex;
                    maxVertexBufferSize *= (uint)vbStride;

                    size = Math.Min(size, maxVertexBufferSize);
                }
            }
            else
            {
                // For non-indexed draws, we can guess the size from the vertex count
                // and stride.

                int firstInstance = (int)_state.State.FirstInstance;

                size = Math.Min(vbSize, (ulong)((firstInstance + firstVertex + vertexCount) * vbStride));
            }

            return size;
        }

        private static ulong Align(ulong size, int alignment)
        {
            ulong align = (ulong)alignment;

            size += align - 1;

            size /= align;
            size *= align;

            return size;
        }
    }
}