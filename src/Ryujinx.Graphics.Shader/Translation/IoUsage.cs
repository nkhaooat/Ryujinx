namespace Ryujinx.Graphics.Shader.Translation
{
    struct IoUsage
    {
        private readonly FeatureFlags _usedFeatures;

        public bool UsesRtLayer => _usedFeatures.HasFlag(FeatureFlags.RtLayer);
        public bool UsesViewportIndex => _usedFeatures.HasFlag(FeatureFlags.ViewportIndex);
        public bool UsesViewportMask => _usedFeatures.HasFlag(FeatureFlags.ViewportMask);
        public byte ClipDistancesWritten { get; }
        public int UserDefinedMap { get; }

        public IoUsage(FeatureFlags usedFeatures, byte clipDistancesWritten, int userDefinedMap)
        {
            _usedFeatures = usedFeatures;
            ClipDistancesWritten = clipDistancesWritten;
            UserDefinedMap = userDefinedMap;
        }

        public IoUsage Combine(IoUsage other)
        {
            return new IoUsage(
                _usedFeatures | other._usedFeatures,
                (byte)(ClipDistancesWritten | other.ClipDistancesWritten),
                UserDefinedMap | other.UserDefinedMap);
        }
    }
}