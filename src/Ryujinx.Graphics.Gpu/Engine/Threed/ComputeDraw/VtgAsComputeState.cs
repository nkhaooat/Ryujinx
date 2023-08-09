using Ryujinx.Common;
using Ryujinx.Common.Logging;
using Ryujinx.Graphics.GAL;
using Ryujinx.Graphics.Gpu.Engine.Types;
using Ryujinx.Graphics.Gpu.Image;
using Ryujinx.Graphics.Gpu.Shader;
using Ryujinx.Graphics.Shader;
using Ryujinx.Graphics.Shader.Translation;
using System;

namespace Ryujinx.Graphics.Gpu.Engine.Threed.ComputeDraw
{
    struct VtgAsComputeState
    {
        private const int ComputeLocalSize = 32;

        private readonly GpuContext _context;
        private readonly GpuChannel _channel;
        private readonly DeviceStateWithShadow<ThreedClassState> _state;
        private readonly VtgAsComputeContext _vacContext;
        private readonly ThreedClass _engine;
        private readonly ShaderAsCompute _vertexAsCompute;
        private readonly ShaderAsCompute _geometryAsCompute;
        private readonly ShaderAsCompute _feedbackAsCompute;
        private readonly IProgram _vertexPassthroughProgram;
        private readonly PrimitiveTopology _topology;
        private readonly int _count;
        private readonly int _instanceCount;
        private readonly int _firstIndex;
        private readonly int _firstVertex;
        private readonly int _firstInstance;
        private readonly bool _indexed;

        private readonly int _vertexDataOffset;
        private readonly int _vertexDataSize;
        private readonly int _geometryVertexDataOffset;
        private readonly int _geometryVertexDataSize;
        private readonly int _geometryIndexDataOffset;
        private readonly int _geometryIndexDataSize;
        private readonly int _geometryIndexDataCount;

        public VtgAsComputeState(
            GpuContext context,
            GpuChannel channel,
            DeviceStateWithShadow<ThreedClassState> state,
            VtgAsComputeContext vacContext,
            ThreedClass engine,
            ShaderAsCompute vertexAsCompute,
            ShaderAsCompute geometryAsCompute,
            ShaderAsCompute feedbackAsCompute,
            IProgram vertexPassthroughProgram,
            PrimitiveTopology topology,
            int count,
            int instanceCount,
            int firstIndex,
            int firstVertex,
            int firstInstance,
            bool indexed)
        {
            _context = context;
            _channel = channel;
            _state = state;
            _vacContext = vacContext;
            _engine = engine;
            _vertexAsCompute = vertexAsCompute;
            _geometryAsCompute = geometryAsCompute;
            _feedbackAsCompute = feedbackAsCompute;
            _vertexPassthroughProgram = vertexPassthroughProgram;
            _topology = topology;
            _count = count;
            _instanceCount = instanceCount;
            _firstIndex = firstIndex;
            _firstVertex = firstVertex;
            _firstInstance = firstInstance;
            _indexed = indexed;

            int vertexDataSize = vertexAsCompute.Reservations.OutputSizeInBytesPerInvocation * count * instanceCount;

            (_vertexDataOffset, _vertexDataSize) = _vacContext.GetVertexDataBuffer(vertexDataSize);

            if (geometryAsCompute != null)
            {
                int totalPrimitivesCount = VtgAsComputeContext.GetPrimitivesCount(topology, count * instanceCount);
                int maxCompleteStrips = GetMaxCompleteStrips(geometryAsCompute.Info.GeometryVerticesPerPrimitive, geometryAsCompute.Info.GeometryMaxOutputVertices);
                int totalVerticesCount = totalPrimitivesCount * geometryAsCompute.Info.GeometryMaxOutputVertices * geometryAsCompute.Info.ThreadsPerInputPrimitive;
                int geometryVbDataSize = totalVerticesCount * geometryAsCompute.Reservations.OutputSizeInBytesPerInvocation;
                int geometryIbDataCount = totalVerticesCount + totalPrimitivesCount * maxCompleteStrips;
                int geometryIbDataSize = geometryIbDataCount * sizeof(uint);

                (_geometryVertexDataOffset, _geometryVertexDataSize) = vacContext.GetGeometryVertexDataBuffer(geometryVbDataSize);
                (_geometryIndexDataOffset, _geometryIndexDataSize) = vacContext.GetGeometryIndexDataBuffer(geometryIbDataSize);

                _geometryIndexDataCount = geometryIbDataCount;
            }
        }

        public readonly void RunVertex()
        {
            _context.Renderer.Pipeline.SetProgram(_vertexAsCompute.HostProgram);

            Span<int> vertexInfo = stackalloc int[8 + 32 * 4 * 2];

            int primitivesCount = VtgAsComputeContext.GetPrimitivesCount(_topology, _count);

            vertexInfo[0] = _count;
            vertexInfo[1] = _instanceCount;
            vertexInfo[2] = _firstVertex;
            vertexInfo[3] = _firstInstance;
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
                    SetDummyBufferTexture(_vertexAsCompute.Reservations, index, format);
                    continue;
                }

                int bufferIndex = vertexAttrib.UnpackBufferIndex();

                GpuVa endAddress = _state.State.VertexBufferEndAddress[bufferIndex];
                var vertexBuffer = _state.State.VertexBufferState[bufferIndex];
                bool instanced = _state.State.VertexBufferInstanced[bufferIndex];

                ulong address = vertexBuffer.Address.Pack();

                if (!vertexBuffer.UnpackEnable() || !_channel.MemoryManager.IsMapped(address))
                {
                    SetDummyBufferTexture(_vertexAsCompute.Reservations, index, format);
                    continue;
                }

                int vbStride = vertexBuffer.UnpackStride();
                ulong vbSize = GetVertexBufferSize(address, endAddress.Pack(), vbStride, _indexed, instanced, _firstVertex, _count);

                ulong oldVbSize = vbSize;

                ulong attributeOffset = (ulong)vertexAttrib.UnpackOffset();
                int componentSize = format.GetScalarSize();

                address += attributeOffset;

                ulong misalign = address & ((ulong)_context.Capabilities.TextureBufferOffsetAlignment - 1);

                vbSize = Align(vbSize - attributeOffset + misalign, componentSize);

                SetBufferTexture(_vertexAsCompute.Reservations, index, format, address - misalign, vbSize);

                vertexInfo[8 + index * 4] = vbStride / componentSize;
                vertexInfo[8 + 32 * 4 + index * 4] = (int)misalign / componentSize;
                vertexInfo[8 + 32 * 4 + index * 4 + 1] = instanced ? vertexBuffer.Divisor : 0;
            }

            if (_indexed)
            {
                SetIndexBufferTexture(_vertexAsCompute.Reservations, _firstIndex, _count, ref vertexInfo[7]);
            }
            else
            {
                SetSequentialIndexBufferTexture(_vertexAsCompute.Reservations, _count);
            }

            int vertexInfoBinding = _vertexAsCompute.Reservations.GetVertexInfoConstantBufferBinding();
            BufferRange vertexInfoRange = new(_vacContext.PushVertexInfo(vertexInfo), 0, vertexInfo.Length * sizeof(int));
            _context.Renderer.Pipeline.SetUniformBuffers(stackalloc[] { new BufferAssignment(vertexInfoBinding, vertexInfoRange) });

            int vertexDataBinding = _vertexAsCompute.Reservations.GetVertexOutputStorageBufferBinding();
            BufferRange vertexDataRange = _vacContext.GetVertexDataBufferRange(_vertexDataOffset, _vertexDataSize);
            _context.Renderer.Pipeline.SetStorageBuffers(stackalloc[] { new BufferAssignment(vertexDataBinding, vertexDataRange) });

            _context.Renderer.Pipeline.DispatchCompute(
                BitUtils.DivRoundUp(_count, ComputeLocalSize),
                BitUtils.DivRoundUp(_instanceCount, ComputeLocalSize),
                1);
        }

        public readonly void RunGeometry()
        {
            if (_geometryAsCompute == null)
            {
                return;
            }

            Span<int> vertexInfo = stackalloc int[8];

            int primitivesCount = VtgAsComputeContext.GetPrimitivesCount(_topology, _count);

            vertexInfo[0] = _count;
            vertexInfo[1] = _instanceCount;
            vertexInfo[2] = _firstVertex;
            vertexInfo[3] = _firstInstance;
            vertexInfo[4] = primitivesCount;

            int vertexInfoBinding = _vertexAsCompute.Reservations.GetVertexInfoConstantBufferBinding();
            BufferRange vertexInfoRange = new(_vacContext.PushVertexInfo(vertexInfo), 0, vertexInfo.Length * sizeof(int));
            _context.Renderer.Pipeline.SetUniformBuffers(stackalloc[] { new BufferAssignment(vertexInfoBinding, vertexInfoRange) });

            int vertexDataBinding = _vertexAsCompute.Reservations.GetVertexOutputStorageBufferBinding();

            // Wait until compute is done.
            // TODO: Batch compute and draw operations to avoid pipeline stalls.
            _context.Renderer.Pipeline.Barrier();
            _context.Renderer.Pipeline.SetProgram(_geometryAsCompute.HostProgram);

            SetTopologyRemapBufferTexture(_geometryAsCompute.Reservations, _topology, _count);

            int geometryVbBinding = _geometryAsCompute.Reservations.GetGeometryVertexOutputStorageBufferBinding();
            int geometryIbBinding = _geometryAsCompute.Reservations.GetGeometryIndexOutputStorageBufferBinding();

            BufferRange vertexDataRange = _vacContext.GetVertexDataBufferRange(_vertexDataOffset, _vertexDataSize);
            BufferRange vertexBuffer = _vacContext.GetGeometryVertexDataBufferRange(_geometryVertexDataOffset, _geometryVertexDataSize);
            BufferRange indexBuffer = _vacContext.GetGeometryIndexDataBufferRange(_geometryIndexDataOffset, _geometryIndexDataSize);

            _context.Renderer.Pipeline.SetStorageBuffers(stackalloc[]
            {
                new BufferAssignment(vertexDataBinding, vertexDataRange),
                new BufferAssignment(geometryVbBinding, vertexBuffer),
                new BufferAssignment(geometryIbBinding, indexBuffer),
            });

            _context.Renderer.Pipeline.DispatchCompute(
                BitUtils.DivRoundUp(primitivesCount, ComputeLocalSize),
                BitUtils.DivRoundUp(_instanceCount, ComputeLocalSize),
                _geometryAsCompute.Info.ThreadsPerInputPrimitive);
        }

        public readonly void RunFragment()
        {
            int vertexDataBinding = _vertexAsCompute.Reservations.GetVertexOutputStorageBufferBinding();

            _context.Renderer.Pipeline.Barrier();

            if (_feedbackAsCompute != null)
            {
                RunFeedback();
            }

            if (!_state.State.RasterizeEnable && !_context.Capabilities.SupportsTransformFeedback)
            {
                // No need to run fragment if rasterizer discard is enabled, and we are emulating transform feedback.
                return;
            }

            if (_geometryAsCompute != null)
            {
                BufferRange vertexBuffer = _vacContext.GetGeometryVertexDataBufferRange(_geometryVertexDataOffset, _geometryVertexDataSize);
                BufferRange indexBuffer = _vacContext.GetGeometryIndexDataBufferRange(_geometryIndexDataOffset, _geometryIndexDataSize);

                _context.Renderer.Pipeline.SetProgram(_vertexPassthroughProgram);
                _context.Renderer.Pipeline.SetIndexBuffer(indexBuffer, IndexType.UInt);
                _context.Renderer.Pipeline.SetStorageBuffers(stackalloc[] { new BufferAssignment(vertexDataBinding, vertexBuffer) });

                _context.Renderer.Pipeline.SetPrimitiveRestart(true, -1);
                _context.Renderer.Pipeline.SetPrimitiveTopology(GetGeometryOutputTopology(_geometryAsCompute.Info.GeometryVerticesPerPrimitive));

                _context.Renderer.Pipeline.DrawIndexed(_geometryIndexDataCount, 1, 0, 0, 0);

                _engine.ForceStateDirtyByIndex(StateUpdater.IndexBufferStateIndex);
                _engine.ForceStateDirtyByIndex(StateUpdater.PrimitiveRestartStateIndex);
            }
            else
            {
                BufferRange vertexDataRange = _vacContext.GetVertexDataBufferRange(_vertexDataOffset, _vertexDataSize);

                _context.Renderer.Pipeline.SetProgram(_vertexPassthroughProgram);
                _context.Renderer.Pipeline.SetStorageBuffers(stackalloc[] { new BufferAssignment(vertexDataBinding, vertexDataRange) });
                _context.Renderer.Pipeline.Draw(_count, _instanceCount, 0, 0);
            }
        }

        private readonly void RunFeedback()
        {
            Span<int> vertexInfo = stackalloc int[8];

            int primitivesCount = VtgAsComputeContext.GetPrimitivesCount(_topology, _count);

            _context.Renderer.Pipeline.SetProgram(_feedbackAsCompute.HostProgram);

            int vertexDataBinding = _feedbackAsCompute.Reservations.GetVertexOutputStorageBufferBinding();
            PrimitiveTopology remapTopology;
            int remapCount;

            if (_geometryAsCompute != null)
            {
                BufferRange vertexBuffer = _vacContext.GetGeometryVertexDataBufferRange(_geometryVertexDataOffset, _geometryVertexDataSize);
                BufferRange indexBuffer = _vacContext.GetGeometryIndexDataBufferRange(_geometryIndexDataOffset, _geometryIndexDataSize);

                _context.Renderer.Pipeline.SetStorageBuffers(stackalloc[] { new BufferAssignment(vertexDataBinding, vertexBuffer) });

                vertexInfo[5] = _geometryIndexDataCount;
                vertexInfo[6] = -1;

                remapTopology = GetGeometryOutputTopology(_geometryAsCompute.Info.GeometryVerticesPerPrimitive);
                remapCount = _geometryAsCompute.Info.GeometryMaxOutputVertices;

                SetIndexBufferTexture(_vertexAsCompute.Reservations, indexBuffer, Format.R32Uint);
            }
            else
            {
                BufferRange vertexDataRange = _vacContext.GetVertexDataBufferRange(_vertexDataOffset, _vertexDataSize);

                _context.Renderer.Pipeline.SetStorageBuffers(stackalloc[] { new BufferAssignment(vertexDataBinding, vertexDataRange) });

                vertexInfo[5] = _count;
                vertexInfo[6] = _state.State.PrimitiveRestartState.Enable ? _state.State.PrimitiveRestartState.Index : -1;

                remapTopology = _topology;
                remapCount = _count;

                if (_indexed)
                {
                    SetIndexBufferTexture(_vertexAsCompute.Reservations, _firstIndex, _count, ref vertexInfo[7]);
                }
                else
                {
                    SetSequentialIndexBufferTexture(_vertexAsCompute.Reservations, _count);
                }
            }

            SetTopologyRemapBufferTexture(_feedbackAsCompute.Reservations, remapTopology, remapCount);

            int vertexInfoBinding = _feedbackAsCompute.Reservations.GetVertexInfoConstantBufferBinding();
            BufferRange vertexInfoRange = new(_vacContext.PushVertexInfo(vertexInfo), 0, vertexInfo.Length * sizeof(int));
            _context.Renderer.Pipeline.SetUniformBuffers(stackalloc[] { new BufferAssignment(vertexInfoBinding, vertexInfoRange) });

            _context.Renderer.Pipeline.DispatchCompute(_geometryAsCompute != null ? 1 : _instanceCount, 1, 1);
        }

        private static PrimitiveTopology GetGeometryOutputTopology(int verticesPerPrimitive)
        {
            return verticesPerPrimitive switch
            {
                3 => PrimitiveTopology.TriangleStrip,
                2 => PrimitiveTopology.LineStrip,
                _ => PrimitiveTopology.Points,
            };
        }

        private static int GetMaxCompleteStrips(int verticesPerPrimitive, int maxOutputVertices)
        {
            return maxOutputVertices / verticesPerPrimitive;
        }

        private readonly void SetDummyBufferTexture(ResourceReservations reservations, int index, Format format)
        {
            ITexture bufferTexture = _vacContext.EnsureBufferTexutre(index + 2, format);
            bufferTexture.SetStorage(_vacContext.GetDummyBufferRange());

            _context.Renderer.Pipeline.SetTextureAndSampler(ShaderStage.Compute, reservations.GetVertexBufferTextureBinding(index), bufferTexture, null);
        }

        private readonly void SetBufferTexture(ResourceReservations reservations, int index, Format format, ulong address, ulong size)
        {
            var memoryManager = _channel.MemoryManager;

            address = memoryManager.Translate(address);
            BufferRange range = memoryManager.Physical.BufferCache.GetBufferRange(address, size);

            ITexture bufferTexture = _vacContext.EnsureBufferTexutre(index + 2, format);
            bufferTexture.SetStorage(range);

            _context.Renderer.Pipeline.SetTextureAndSampler(ShaderStage.Compute, reservations.GetVertexBufferTextureBinding(index), bufferTexture, null);
        }

        private readonly void SetIndexBufferTexture(ResourceReservations reservations, int firstIndex, int count, ref int misalignedOffset)
        {
            ulong address = _state.State.IndexBufferState.Address.Pack();
            ulong indexOffset = (ulong)firstIndex;
            ulong size = (ulong)count;

            int shift = 0;
            Format format = Format.R8Uint;

            switch (_state.State.IndexBufferState.Type)
            {
                case IndexType.UShort:
                    shift = 1;
                    format = Format.R16Uint;
                    break;
                case IndexType.UInt:
                    shift = 2;
                    format = Format.R32Uint;
                    break;
            }

            indexOffset <<= shift;
            size <<= shift;

            var memoryManager = _channel.MemoryManager;

            address = memoryManager.Translate(address + indexOffset);
            ulong misalign = address & ((ulong)_context.Capabilities.TextureBufferOffsetAlignment - 1);
            BufferRange range = memoryManager.Physical.BufferCache.GetBufferRange(address - misalign, size + misalign);
            misalignedOffset = (int)misalign >> shift;

            SetIndexBufferTexture(reservations, range, format);
        }

        private readonly void SetIndexBufferTexture(ResourceReservations reservations, BufferRange range, Format format)
        {
            ITexture bufferTexture = _vacContext.EnsureBufferTexutre(0, format);
            bufferTexture.SetStorage(range);

            _context.Renderer.Pipeline.SetTextureAndSampler(ShaderStage.Compute, reservations.GetIndexBufferTextureBinding(), bufferTexture, null);
        }

        private readonly void SetTopologyRemapBufferTexture(ResourceReservations reservations, PrimitiveTopology topology, int count)
        {
            ITexture bufferTexture = _vacContext.EnsureBufferTexutre(1, Format.R32Uint);
            bufferTexture.SetStorage(_vacContext.GetOrCreateTopologyRemapBuffer(topology, count));

            _context.Renderer.Pipeline.SetTextureAndSampler(ShaderStage.Compute, reservations.GetTopologyRemapBufferTextureBinding(), bufferTexture, null);
        }

        private readonly void SetSequentialIndexBufferTexture(ResourceReservations reservations, int count)
        {
            BufferHandle sequentialIndexBuffer = _vacContext.GetSequentialIndexBuffer(count);

            ITexture bufferTexture = _vacContext.EnsureBufferTexutre(0, Format.R32Uint);
            bufferTexture.SetStorage(new BufferRange(sequentialIndexBuffer, 0, count * sizeof(uint)));

            _context.Renderer.Pipeline.SetTextureAndSampler(ShaderStage.Compute, reservations.GetIndexBufferTextureBinding(), bufferTexture, null);
        }

        private readonly ulong GetVertexBufferSize(ulong vbAddress, ulong vbEndAddress, int vbStride, bool indexed, bool instanced, int firstVertex, int vertexCount)
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