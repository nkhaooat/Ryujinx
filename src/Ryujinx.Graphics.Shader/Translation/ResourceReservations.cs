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
        public int InputSizePerInvocation { get; }
        public int OutputSizePerInvocation { get; }
        public int OutputSizeInBytesPerInvocation => OutputSizePerInvocation * sizeof(uint);

        private readonly int _tfeInfoSbBinding;
        private readonly int _tfeBufferSbBaseBinding;
        private readonly int _vertexInfoCbBinding;
        private readonly int _vertexOutputSbBinding;
        private readonly int _geometryVbOutputSbBinding;
        private readonly int _geometryIbOutputSbBinding;
        private readonly int _indexBufferTextureBinding;
        private readonly int _topologyRemapBufferTextureBinding;
        private readonly int _vertexBufferTextureBaseBinding;

        private readonly Dictionary<IoDefinition, int> _offsets;
        internal IReadOnlyDictionary<IoDefinition, int> Offsets => _offsets;

        internal ResourceReservations(
            IGpuAccessor gpuAccessor,
            bool isTransformFeedbackEmulated,
            bool vertexAsCompute,
            IoUsage? vacInput,
            IoUsage vacOutput)
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

                // One storage buffer for the output geometry vertex data.
                _geometryVbOutputSbBinding = ReservedStorageBuffers++;

                // One storage buffer for the output geometry index data.
                _geometryIbOutputSbBinding = ReservedStorageBuffers++;

                // Enough textures reserved for all vertex attributes, plus the index buffer.
                _indexBufferTextureBinding = ReservedTextures;
                _topologyRemapBufferTextureBinding = ReservedTextures + 1;
                _vertexBufferTextureBaseBinding = ReservedTextures + 2;
                ReservedTextures += 2 + MaxVertexBufferTextures;
            }

            if (vertexAsCompute)
            {
                _offsets = new();

                if (vacInput.HasValue)
                {
                    InputSizePerInvocation = FillIoOffsetMap(gpuAccessor, StorageKind.Input, vacInput.Value);
                }

                OutputSizePerInvocation = FillIoOffsetMap(gpuAccessor, StorageKind.Output, vacOutput);
            }
        }

        private int FillIoOffsetMap(IGpuAccessor gpuAccessor, StorageKind storageKind, IoUsage vacUsage)
        {
            int offset = 0;

            for (int c = 0; c < 4; c++)
            {
                _offsets.Add(new IoDefinition(storageKind, IoVariable.Position, 0, c), offset++);
            }

            _offsets.Add(new IoDefinition(storageKind, IoVariable.PointSize), offset++);

            int clipDistancesWrittenMap = vacUsage.ClipDistancesWritten;

            while (clipDistancesWrittenMap != 0)
            {
                int index = BitOperations.TrailingZeroCount(clipDistancesWrittenMap);

                _offsets.Add(new IoDefinition(storageKind, IoVariable.ClipDistance, 0, index), offset++);

                clipDistancesWrittenMap &= ~(1 << index);
            }

            if (vacUsage.UsesRtLayer)
            {
                _offsets.Add(new IoDefinition(storageKind, IoVariable.Layer), offset++);
            }

            if (vacUsage.UsesViewportIndex && gpuAccessor.QueryHostSupportsViewportIndexVertexTessellation())
            {
                _offsets.Add(new IoDefinition(storageKind, IoVariable.VertexIndex), offset++);
            }

            if (vacUsage.UsesViewportMask && gpuAccessor.QueryHostSupportsViewportMask())
            {
                _offsets.Add(new IoDefinition(storageKind, IoVariable.ViewportMask), offset++);
            }

            int usedDefinedMap = vacUsage.UserDefinedMap;

            while (usedDefinedMap != 0)
            {
                int location = BitOperations.TrailingZeroCount(usedDefinedMap);

                for (int c = 0; c < 4; c++)
                {
                    _offsets.Add(new IoDefinition(storageKind, IoVariable.UserDefined, location, c), offset++);
                }

                usedDefinedMap &= ~(1 << location);
            }

            return offset;
        }

        internal static bool IsVectorOrArrayVariable(IoVariable variable)
        {
            return variable switch
            {
                IoVariable.ClipDistance or
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

        public int GetGeometryVertexOutputStorageBufferBinding()
        {
            return _geometryVbOutputSbBinding;
        }

        public int GetGeometryIndexOutputStorageBufferBinding()
        {
            return _geometryIbOutputSbBinding;
        }

        public int GetIndexBufferTextureBinding()
        {
            return _indexBufferTextureBinding;
        }

        public int GetTopologyRemapBufferTextureBinding()
        {
            return _topologyRemapBufferTextureBinding;
        }

        public int GetVertexBufferTextureBinding(int vaLocation)
        {
            return _vertexBufferTextureBaseBinding + vaLocation;
        }

        internal bool TryGetOffset(StorageKind storageKind, int location, int component, out int offset)
        {
            return _offsets.TryGetValue(new IoDefinition(storageKind, IoVariable.UserDefined, location, component), out offset);
        }

        internal bool TryGetOffset(StorageKind storageKind, IoVariable ioVariable, int component, out int offset)
        {
            return _offsets.TryGetValue(new IoDefinition(storageKind, ioVariable, 0, component), out offset);
        }

        internal bool TryGetOffset(StorageKind storageKind, IoVariable ioVariable, out int offset)
        {
            return _offsets.TryGetValue(new IoDefinition(storageKind, ioVariable, 0, 0), out offset);
        }
    }
}