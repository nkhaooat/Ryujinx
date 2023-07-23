using Ryujinx.Graphics.Shader.IntermediateRepresentation;
using Ryujinx.Graphics.Shader.StructuredIr;
using System.Collections.Generic;
using System.Numerics;

namespace Ryujinx.Graphics.Shader.Translation
{
    public class ResourceReservations
    {
        public const int TfeBuffersCount = 4;

        public const int MaxVertexBufferTextures = 32;

        public int ReservedConstantBuffers { get; }
        public int ReservedStorageBuffers { get; }
        public int ReservedTextures { get; }
        public int ReservedImages { get; }
        public int OutputSizePerInvocation { get; }
        public int OutputSizeInBytesPerInvocation => OutputSizePerInvocation * sizeof(uint);

        private readonly int _tfeInfoSbBinding;
        private readonly int _tfeBufferSbBaseBinding;
        private readonly int _vertexInfoCbBinding;
        private readonly int _vertexOutputSbBinding;
        private readonly int _indexBufferTextureBinding;
        private readonly int _vertexBufferTextureBaseBinding;

        private readonly Dictionary<IoDefinition, int> _outputOffsets;

        internal IReadOnlyDictionary<IoDefinition, int> OutputOffsets => _outputOffsets;

        internal ResourceReservations(bool isTransformFeedbackEmulated, bool vertexAsCompute, int vacOutputMap)
        {
            // All stages reserves the first constant buffer binding for the support buffer.
            ReservedConstantBuffers = 1;
            ReservedStorageBuffers = 0;
            ReservedTextures = 0;
            ReservedImages = 0;

            if (isTransformFeedbackEmulated)
            {
                // Transform feedback emulation currently always uses 5 storage buffers.
                _tfeInfoSbBinding = ReservedStorageBuffers;
                _tfeBufferSbBaseBinding = ReservedStorageBuffers + 1;
                ReservedStorageBuffers = 1 + TfeBuffersCount;
            }

            if (vertexAsCompute)
            {
                // One constant buffer reserved for vertex related state.
                _vertexInfoCbBinding = ReservedConstantBuffers++;

                // One storage buffer for the output vertex data.
                _vertexOutputSbBinding = ReservedStorageBuffers++;

                // Enough textures reserved for all vertex attributes, plus the index buffer.
                _indexBufferTextureBinding = ReservedTextures;
                _vertexBufferTextureBaseBinding = ReservedTextures + 1;
                ReservedTextures += 1 + MaxVertexBufferTextures;
            }

            _outputOffsets = new();

            if (vertexAsCompute)
            {
                int offset = 0;

                for (int c = 0; c < 4; c++)
                {
                    _outputOffsets.Add(new IoDefinition(StorageKind.Output, IoVariable.Position, 0, c), offset++);
                }

                _outputOffsets.Add(new IoDefinition(StorageKind.Output, IoVariable.PointSize), offset++);

                while (vacOutputMap != 0)
                {
                    int location = BitOperations.TrailingZeroCount(vacOutputMap);

                    for (int c = 0; c < 4; c++)
                    {
                        _outputOffsets.Add(new IoDefinition(StorageKind.Output, IoVariable.UserDefined, location, c), offset++);
                    }

                    vacOutputMap &= ~(1 << location);
                }

                OutputSizePerInvocation = offset;
            }
        }

        internal static bool IsVectorVariable(IoVariable variable)
        {
            return variable switch
            {
                IoVariable.Position => true,
                _ => false,
            };
        }

        public int GetTfeInfoStorageBufferBinding()
        {
            return _tfeInfoSbBinding;
        }

        public int GetTfeBufferStorageBufferBinding(int bufferIndex)
        {
            return _tfeBufferSbBaseBinding + bufferIndex;
        }

        public int GetVertexInfoConstantBufferBinding()
        {
            return _vertexInfoCbBinding;
        }

        public int GetVertexOutputStorageBufferBinding()
        {
            return _vertexOutputSbBinding;
        }

        public int GetIndexBufferTextureBinding()
        {
            return _indexBufferTextureBinding;
        }

        public int GetVertexBufferTextureBinding(int vaLocation)
        {
            return _vertexBufferTextureBaseBinding + vaLocation;
        }

        internal bool TryGetOutputOffset(int location, int component, out int offset)
        {
            return _outputOffsets.TryGetValue(new IoDefinition(StorageKind.Output, IoVariable.UserDefined, location, component), out offset);
        }

        internal bool TryGetOutputOffset(IoVariable ioVariable, int component, out int offset)
        {
            return _outputOffsets.TryGetValue(new IoDefinition(StorageKind.Output, ioVariable, 0, component), out offset);
        }

        internal bool TryGetOutputOffset(IoVariable ioVariable, out int offset)
        {
            return _outputOffsets.TryGetValue(new IoDefinition(StorageKind.Output, ioVariable, 0, 0), out offset);
        }
    }
}