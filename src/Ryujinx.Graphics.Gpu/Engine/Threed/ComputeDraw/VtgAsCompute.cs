using Ryujinx.Graphics.GAL;
using Ryujinx.Graphics.Gpu.Shader;

namespace Ryujinx.Graphics.Gpu.Engine.Threed.ComputeDraw
{
    class VtgAsCompute
    {
        private readonly GpuContext _context;
        private readonly GpuChannel _channel;
        private readonly DeviceStateWithShadow<ThreedClassState> _state;
        private readonly VtgAsComputeContext _vacContext;

        public VtgAsCompute(GpuContext context, GpuChannel channel, DeviceStateWithShadow<ThreedClassState> state)
        {
            _context = context;
            _channel = channel;
            _state = state;
            _vacContext = new(context);
        }

        public void DrawAsCompute(
            ThreedClass engine,
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
            VtgAsComputeState state = new(
                _context,
                _channel,
                _state,
                _vacContext,
                engine,
                vertexAsCompute,
                geometryAsCompute,
                vertexPassthroughProgram,
                topology,
                count,
                instanceCount,
                firstIndex,
                firstVertex,
                firstInstance,
                indexed);

            state.RunVertex();
            state.RunGeometry();
            state.RunFragment();

            _vacContext.FreeBuffers();
        }
    }
}