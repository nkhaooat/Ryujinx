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

        private BufferTextureCache[] _bufferTextures;
        private BufferHandle _dummyBuffer;
        private BufferHandle _vertexInfoBuffer;
        private BufferHandle _vertexDataBuffer;
        private int _vertexDataBufferSize;
        private BufferHandle _sequentialIndexBuffer;
        private int _sequentialIndexBufferCount;

        public VtgAsCompute(GpuContext context, GpuChannel channel, DeviceStateWithShadow<ThreedClassState> state)
        {
            _context = context;
            _channel = channel;
            _state = state;
            _bufferTextures = new BufferTextureCache[Constants.TotalVertexBuffers + 1];
        }

        public void DrawAsCompute(
            ShaderAsCompute computeProgram,
            IProgram vertexPassthroughProgram,
            int count,
            int instanceCount,
            int firstIndex,
            int firstVertex,
            int firstInstance,
            bool indexed)
        {
            _context.Renderer.Pipeline.SetProgram(computeProgram.HostProgram);

            Span<int> vertexInfo = stackalloc int[4 + 4 * Constants.TotalVertexAttribs];

            vertexInfo[0] = count;
            vertexInfo[1] = instanceCount;
            vertexInfo[2] = firstVertex;
            vertexInfo[3] = firstInstance;

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
                    vertexInfo[4 + index * 4 + c] = -1;
                }

                if (vertexAttrib.UnpackIsConstant())
                {
                    SetDummyBufferTexture(computeProgram.Reservations, index, format);
                    continue;
                }

                int bufferIndex = vertexAttrib.UnpackBufferIndex();

                GpuVa endAddress = _state.State.VertexBufferEndAddress[bufferIndex];
                var vertexBuffer = _state.State.VertexBufferState[bufferIndex];
                bool instanced = _state.State.VertexBufferInstanced[bufferIndex];

                ulong address = vertexBuffer.Address.Pack();

                if (!vertexBuffer.UnpackEnable() || !_channel.MemoryManager.IsMapped(address))
                {
                    SetDummyBufferTexture(computeProgram.Reservations, index, format);
                    continue;
                }

                int vbStride = vertexBuffer.UnpackStride();
                ulong vbSize = GetVertexBufferSize(address, endAddress.Pack(), vbStride, indexed, instanced, firstVertex, count);

                ulong oldVbSize = vbSize;

                ulong attributeOffset = (ulong)firstVertex * (ulong)vbStride + (ulong)vertexAttrib.UnpackOffset();
                int componentSize = format.GetScalarSize();

                address += attributeOffset;
                vbSize = Align(vbSize - attributeOffset, componentSize);

                SetBufferTexture(computeProgram.Reservations, index, format, address, vbSize);

                vertexInfo[4 + index * 4] = vbStride / componentSize;
            }

            if (indexed)
            {
                SetIndexBufferTexture(computeProgram.Reservations, firstIndex, count);
            }
            else
            {
                SetSequentialIndexBufferTexture(computeProgram.Reservations, count);
            }

            int vertexInfoBinding = computeProgram.Reservations.GetVertexInfoConstantBufferBinding();
            BufferRange vertexInfoRange = new(PushVertexInfo(vertexInfo), 0, vertexInfo.Length * sizeof(int));
            _context.Renderer.Pipeline.SetUniformBuffers(stackalloc[] { new BufferAssignment(vertexInfoBinding, vertexInfoRange) });

            int vertexDataBinding = computeProgram.Reservations.GetVertexOutputStorageBufferBinding();
            int vertexDataSize = computeProgram.Reservations.OutputSizeInBytesPerInvocation * count * instanceCount;
            BufferRange vertexDataRange = new(EnsureVertexDataBuffer(vertexDataSize), 0, vertexDataSize);
            _context.Renderer.Pipeline.SetStorageBuffers(stackalloc[] { new BufferAssignment(vertexDataBinding, vertexDataRange) });

            _context.Renderer.Pipeline.DispatchCompute(count, instanceCount, 1);

            // Wait until compute is done.
            // TODO: Batch compute and draw operations to avoid pipeline stalls.
            _context.Renderer.Pipeline.Barrier();

            _context.Renderer.Pipeline.SetProgram(vertexPassthroughProgram);

            _context.Renderer.Pipeline.Draw(count * instanceCount, 1, 0, 0);
        }

        private void SetDummyBufferTexture(ResourceReservations reservations, int index, Format format)
        {
            if (_dummyBuffer == BufferHandle.Null)
            {
                _dummyBuffer = _context.Renderer.CreateBuffer(DummyBufferSize);
                _context.Renderer.Pipeline.ClearBuffer(_dummyBuffer, 0, DummyBufferSize, 0);
            }

            ITexture bufferTexture = EnsureBufferTexutre(index + 1, format);
            bufferTexture.SetStorage(new BufferRange(_dummyBuffer, 0, DummyBufferSize));

            _context.Renderer.Pipeline.SetTextureAndSampler(ShaderStage.Compute, reservations.GetVertexBufferTextureBinding(index), bufferTexture, null);
        }

        private void SetBufferTexture(ResourceReservations reservations, int index, Format format, ulong address, ulong size)
        {
            var memoryManager = _channel.MemoryManager;

            address = memoryManager.Translate(address);
            BufferRange range = memoryManager.Physical.BufferCache.GetBufferRange(address, size);

            ITexture bufferTexture = EnsureBufferTexutre(index + 1, format);
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

        private BufferHandle EnsureVertexDataBuffer(int size)
        {
            if (_vertexDataBufferSize < size)
            {
                if (_vertexDataBuffer != BufferHandle.Null)
                {
                    _context.Renderer.DeleteBuffer(_vertexDataBuffer);
                }

                _vertexDataBuffer = _context.Renderer.CreateBuffer(size);
                _vertexDataBufferSize = size;
            }

            return _vertexDataBuffer;
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