namespace Ryujinx.Graphics.Shader.Translation
{
    struct IoUsage
    {
        public bool UsesRtLayer { get; }
        public bool UsesViewportIndex { get; }
        public bool UsesViewportMask { get; }
        public byte ClipDistancesWritten { get; }
        public int UserDefinedMap { get; }

        public IoUsage(FeatureFlags usedFeatures, byte clipDistancesWritten, int userDefinedMap)
        {
            UsesRtLayer = usedFeatures.HasFlag(FeatureFlags.RtLayer);
            UsesViewportIndex = usedFeatures.HasFlag(FeatureFlags.ViewportIndex);
            UsesViewportMask = usedFeatures.HasFlag(FeatureFlags.ViewportMask);
            ClipDistancesWritten = clipDistancesWritten;
            UserDefinedMap = userDefinedMap;
        }
    }
}