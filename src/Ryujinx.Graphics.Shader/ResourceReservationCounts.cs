namespace Ryujinx.Graphics.Shader
{
    public struct ResourceReservationCounts
    {
        private const int TfeBuffersCount = 4;

        private const int MaxVertexBufferTextures = 32;

        public int ReservedConstantBuffers { get; }
        public int ReservedStorageBuffers { get; }
        public int ReservedTextures { get; }
        public int ReservedImages { get; }

        public ResourceReservationCounts(bool isTransformFeedbackEmulated, bool vertexAsCompute)
        {
            // All stages reserves the first constant buffer binding for the support buffer.
            ReservedConstantBuffers = 1;
            ReservedStorageBuffers = 0;
            ReservedTextures = 0;
            ReservedImages = 0;

            if (isTransformFeedbackEmulated)
            {
                // Transform feedback emulation currently always uses 4 storage buffers.
                ReservedStorageBuffers = TfeBuffersCount;
            }

            if (vertexAsCompute)
            {
                // One constant buffer reserved for vertex related state.
                ReservedConstantBuffers++;

                // One storage buffer for the output vertex data, two for geometry output vertex and index data.
                ReservedStorageBuffers += 3;

                // Enough textures reserved for all vertex attributes, plus the index and topology remap buffers.
                ReservedTextures += 2 + MaxVertexBufferTextures;
            }
        }
    }
}
