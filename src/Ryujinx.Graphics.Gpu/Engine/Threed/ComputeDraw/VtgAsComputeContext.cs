using Ryujinx.Common;
using Ryujinx.Graphics.GAL;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace Ryujinx.Graphics.Gpu.Engine.Threed.ComputeDraw
{
    class VtgAsComputeContext
    {
        private const int DummyBufferSize = 16;

        private readonly GpuContext _context;

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
            public int Offset;
            public int Size;
        }

        private readonly struct IndexBuffer
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

            public readonly BufferRange ToRange()
            {
                return new BufferRange(Handle, 0, Size);
            }

            public readonly BufferRange ToRange(int size)
            {
                return new BufferRange(Handle, 0, size);
            }
        }

        private readonly BufferTextureCache[] _bufferTextures;
        private BufferHandle _dummyBuffer;
        private Buffer _vertexDataBuffer;
        private Buffer _geometryVertexDataBuffer;
        private Buffer _geometryIndexDataBuffer;
        private BufferHandle _sequentialIndexBuffer;
        private int _sequentialIndexBufferCount;

        private readonly Dictionary<PrimitiveTopology, IndexBuffer> _topologyRemapBuffers;

        public VertexInfoBufferUpdater VertexInfoBufferUpdater { get; }

        public VtgAsComputeContext(GpuContext context)
        {
            _context = context;
            _bufferTextures = new BufferTextureCache[Constants.TotalVertexBuffers + 2];
            _topologyRemapBuffers = new();
            VertexInfoBufferUpdater = new(context.Renderer);
        }

        public static int GetPrimitivesCount(PrimitiveTopology primitiveType, int count)
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

        public BufferRange GetOrCreateTopologyRemapBuffer(PrimitiveTopology topology, int count)
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
                    data[^1] = 0;

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

        public ITexture EnsureBufferTexutre(int index, Format format)
        {
            return (_bufferTextures[index] ??= new()).Get(_context.Renderer, format);
        }

        public (int, int) GetVertexDataBuffer(int size)
        {
            return EnsureBuffer(ref _vertexDataBuffer, size);
        }

        public (int, int) GetGeometryVertexDataBuffer(int size)
        {
            return EnsureBuffer(ref _geometryVertexDataBuffer, size);
        }

        public (int, int) GetGeometryIndexDataBuffer(int size)
        {
            return EnsureBuffer(ref _geometryIndexDataBuffer, size);
        }

        public BufferRange GetVertexDataBufferRange(int offset, int size)
        {
            return new BufferRange(_vertexDataBuffer.Handle, offset, size);
        }

        public BufferRange GetGeometryVertexDataBufferRange(int offset, int size)
        {
            return new BufferRange(_geometryVertexDataBuffer.Handle, offset, size);
        }

        public BufferRange GetGeometryIndexDataBufferRange(int offset, int size)
        {
            return new BufferRange(_geometryIndexDataBuffer.Handle, offset, size);
        }

        public BufferRange GetDummyBufferRange()
        {
            if (_dummyBuffer == BufferHandle.Null)
            {
                _dummyBuffer = _context.Renderer.CreateBuffer(DummyBufferSize);
                _context.Renderer.Pipeline.ClearBuffer(_dummyBuffer, 0, DummyBufferSize, 0);
            }

            return new BufferRange(_dummyBuffer, 0, DummyBufferSize);
        }

        public BufferHandle GetSequentialIndexBuffer(int count)
        {
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

            return _sequentialIndexBuffer;
        }

        private (int, int) EnsureBuffer(ref Buffer buffer, int size)
        {
            int newSize = buffer.Offset + size;

            if (buffer.Size < newSize)
            {
                if (buffer.Handle != BufferHandle.Null)
                {
                    _context.Renderer.DeleteBuffer(buffer.Handle);
                }

                buffer.Handle = _context.Renderer.CreateBuffer(newSize);
                buffer.Size = newSize;
            }

            int offset = buffer.Offset;

            buffer.Offset = BitUtils.AlignUp(newSize, _context.Capabilities.StorageBufferOffsetAlignment);

            return (offset, size);
        }

        public void FreeBuffers()
        {
            _vertexDataBuffer.Offset = 0;
            _geometryVertexDataBuffer.Offset = 0;
            _geometryIndexDataBuffer.Offset = 0;
        }
    }
}
