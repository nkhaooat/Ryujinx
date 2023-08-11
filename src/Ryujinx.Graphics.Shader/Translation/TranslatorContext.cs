﻿using Ryujinx.Graphics.Shader.CodeGen;
using Ryujinx.Graphics.Shader.CodeGen.Glsl;
using Ryujinx.Graphics.Shader.CodeGen.Spirv;
using Ryujinx.Graphics.Shader.Decoders;
using Ryujinx.Graphics.Shader.IntermediateRepresentation;
using Ryujinx.Graphics.Shader.StructuredIr;
using Ryujinx.Graphics.Shader.Translation.Optimizations;
using Ryujinx.Graphics.Shader.Translation.Transforms;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using static Ryujinx.Graphics.Shader.IntermediateRepresentation.OperandHelper;
using static Ryujinx.Graphics.Shader.Translation.Translator;

namespace Ryujinx.Graphics.Shader.Translation
{
    public class TranslatorContext
    {
        private readonly DecodedProgram _program;
        private readonly int _localMemorySize;
        private IoUsage _vertexOutput;

        public ulong Address { get; }
        public int Size { get; }
        public int Cb1DataSize => _program.Cb1DataSize;

        internal AttributeUsage AttributeUsage => _program.AttributeUsage;

        internal ShaderDefinitions Definitions { get; }

        public ShaderStage Stage => Definitions.Stage;

        internal IGpuAccessor GpuAccessor { get; }

        internal TranslationOptions Options { get; }

        internal FeatureFlags UsedFeatures { get; private set; }

        public bool LayerOutputWritten { get; private set; }
        public int LayerOutputAttribute { get; private set; }

        internal TranslatorContext(
            ulong address,
            int size,
            int localMemorySize,
            ShaderDefinitions definitions,
            IGpuAccessor gpuAccessor,
            TranslationOptions options,
            DecodedProgram program)
        {
            Address = address;
            Size = size;
            _program = program;
            _localMemorySize = localMemorySize;
            _vertexOutput = new IoUsage(FeatureFlags.None, 0, -1);
            Definitions = definitions;
            GpuAccessor = gpuAccessor;
            Options = options;
            UsedFeatures = program.UsedFeatures;
        }

        private static bool IsLoadUserDefined(Operation operation)
        {
            // TODO: Check if sources count match and all sources are constant.
            return operation.Inst == Instruction.Load && (IoVariable)operation.GetSource(0).Value == IoVariable.UserDefined;
        }

        private static bool IsStoreUserDefined(Operation operation)
        {
            // TODO: Check if sources count match and all sources are constant.
            return operation.Inst == Instruction.Store && (IoVariable)operation.GetSource(0).Value == IoVariable.UserDefined;
        }

        private static FunctionCode[] Combine(FunctionCode[] a, FunctionCode[] b, int aStart)
        {
            // Here we combine two shaders.
            // For shader A:
            // - All user attribute stores on shader A are turned into copies to a
            // temporary variable. It's assumed that shader B will consume them.
            // - All return instructions are turned into branch instructions, the
            // branch target being the start of the shader B code.
            // For shader B:
            // - All user attribute loads on shader B are turned into copies from a
            // temporary variable, as long that attribute is written by shader A.
            FunctionCode[] output = new FunctionCode[a.Length + b.Length - 1];

            List<Operation> ops = new(a.Length + b.Length);

            Operand[] temps = new Operand[AttributeConsts.UserAttributesCount * 4];

            Operand lblB = Label();

            for (int index = aStart; index < a[0].Code.Length; index++)
            {
                Operation operation = a[0].Code[index];

                if (IsStoreUserDefined(operation))
                {
                    int tIndex = operation.GetSource(1).Value * 4 + operation.GetSource(2).Value;

                    Operand temp = temps[tIndex];

                    if (temp == null)
                    {
                        temp = Local();

                        temps[tIndex] = temp;
                    }

                    operation.Dest = temp;
                    operation.TurnIntoCopy(operation.GetSource(operation.SourcesCount - 1));
                }

                if (operation.Inst == Instruction.Return)
                {
                    ops.Add(new Operation(Instruction.Branch, lblB));
                }
                else
                {
                    ops.Add(operation);
                }
            }

            ops.Add(new Operation(Instruction.MarkLabel, lblB));

            for (int index = 0; index < b[0].Code.Length; index++)
            {
                Operation operation = b[0].Code[index];

                if (IsLoadUserDefined(operation))
                {
                    int tIndex = operation.GetSource(1).Value * 4 + operation.GetSource(2).Value;

                    Operand temp = temps[tIndex];

                    if (temp != null)
                    {
                        operation.TurnIntoCopy(temp);
                    }
                }

                ops.Add(operation);
            }

            output[0] = new FunctionCode(ops.ToArray());

            for (int i = 1; i < a.Length; i++)
            {
                output[i] = a[i];
            }

            for (int i = 1; i < b.Length; i++)
            {
                output[a.Length + i - 1] = b[i];
            }

            return output;
        }

        internal int GetDepthRegister()
        {
            // The depth register is always two registers after the last color output.
            return BitOperations.PopCount((uint)Definitions.OmapTargets) + 1;
        }

        public void SetLayerOutputAttribute(int attr)
        {
            LayerOutputWritten = true;
            LayerOutputAttribute = attr;
        }

        public void SetLastInVertexPipeline()
        {
            Definitions.LastInVertexPipeline = true;
        }

        public void SetNextStage(TranslatorContext nextStage)
        {
            AttributeUsage.MergeFromtNextStage(
                Definitions.GpPassthrough,
                nextStage.UsedFeatures.HasFlag(FeatureFlags.FixedFuncAttr),
                nextStage.AttributeUsage);

            // We don't consider geometry shaders using the geometry shader passthrough feature
            // as being the last because when this feature is used, it can't actually modify any of the outputs,
            // so the stage that comes before it is the last one that can do modifications.
            if (nextStage.Definitions.Stage != ShaderStage.Fragment &&
                (nextStage.Definitions.Stage != ShaderStage.Geometry || !nextStage.Definitions.GpPassthrough))
            {
                Definitions.LastInVertexPipeline = false;
            }
        }

        public ShaderProgram Translate(bool asCompute = false)
        {
            if (asCompute)
            {
                // TODO: Stop doing this here and pass used features to the emitter context.
                UsedFeatures |= FeatureFlags.VtgAsCompute;
            }

            ResourceManager resourceManager = CreateResourceManager(asCompute);

            bool usesLocalMemory = _program.UsedFeatures.HasFlag(FeatureFlags.LocalMemory);

            resourceManager.SetCurrentLocalMemory(_localMemorySize, usesLocalMemory);

            if (Stage == ShaderStage.Compute)
            {
                bool usesSharedMemory = _program.UsedFeatures.HasFlag(FeatureFlags.SharedMemory);

                resourceManager.SetCurrentSharedMemory(GpuAccessor.QueryComputeSharedMemorySize(), usesSharedMemory);
            }

            FunctionCode[] code = EmitShader(this, resourceManager, _program, initializeOutputs: true, out _);

            return Translate(code, resourceManager, GetUsedFeatures(asCompute), _program.ClipDistancesWritten, asCompute);
        }

        public ShaderProgram Translate(TranslatorContext other, bool asCompute = false)
        {
            if (asCompute)
            {
                // TODO: Stop doing this here and pass used features to the emitter context.
                UsedFeatures |= FeatureFlags.VtgAsCompute;
            }

            ResourceManager resourceManager = CreateResourceManager(asCompute);

            bool usesLocalMemory = _program.UsedFeatures.HasFlag(FeatureFlags.LocalMemory);
            resourceManager.SetCurrentLocalMemory(_localMemorySize, usesLocalMemory);

            FunctionCode[] code = EmitShader(this, resourceManager, _program, initializeOutputs: false, out _);

            bool otherUsesLocalMemory = other._program.UsedFeatures.HasFlag(FeatureFlags.LocalMemory);
            resourceManager.SetCurrentLocalMemory(other._localMemorySize, otherUsesLocalMemory);

            FunctionCode[] otherCode = EmitShader(other, resourceManager, other._program, initializeOutputs: true, out int aStart);

            code = Combine(otherCode, code, aStart);

            return Translate(
                code,
                resourceManager,
                GetUsedFeatures(asCompute) | other.UsedFeatures,
                (byte)(_program.ClipDistancesWritten | other._program.ClipDistancesWritten),
                asCompute);
        }

        private ShaderProgram Translate(FunctionCode[] functions, ResourceManager resourceManager, FeatureFlags usedFeatures, byte clipDistancesWritten, bool asCompute)
        {
            var cfgs = new ControlFlowGraph[functions.Length];
            var frus = new RegisterUsage.FunctionRegisterUsage[functions.Length];

            for (int i = 0; i < functions.Length; i++)
            {
                cfgs[i] = ControlFlowGraph.Create(functions[i].Code);

                if (i != 0)
                {
                    frus[i] = RegisterUsage.RunPass(cfgs[i]);
                }
            }

            List<Function> funcs = new(functions.Length);

            for (int i = 0; i < functions.Length; i++)
            {
                funcs.Add(null);
            }

            HelperFunctionManager hfm = new(funcs, Definitions.Stage);

            for (int i = 0; i < functions.Length; i++)
            {
                var cfg = cfgs[i];

                int inArgumentsCount = 0;
                int outArgumentsCount = 0;

                if (i != 0)
                {
                    var fru = frus[i];

                    inArgumentsCount = fru.InArguments.Length;
                    outArgumentsCount = fru.OutArguments.Length;
                }

                if (cfg.Blocks.Length != 0)
                {
                    RegisterUsage.FixupCalls(cfg.Blocks, frus);

                    Dominance.FindDominators(cfg);
                    Dominance.FindDominanceFrontiers(cfg.Blocks);

                    Ssa.Rename(cfg.Blocks);

                    TransformContext context = new(
                        hfm,
                        cfg.Blocks,
                        Definitions,
                        resourceManager,
                        GpuAccessor,
                        Options.TargetLanguage,
                        Definitions.Stage,
                        ref usedFeatures);

                    Optimizer.RunPass(context);
                    TransformPasses.RunPass(context);
                }

                funcs[i] = new Function(cfg.Blocks, $"fun{i}", false, inArgumentsCount, outArgumentsCount);
            }

            var identification = ShaderIdentifier.Identify(funcs, GpuAccessor, Definitions.Stage, Definitions.InputTopology, out int layerInputAttr);

            return Generate(
                funcs,
                AttributeUsage,
                GetDefinitions(asCompute),
                Definitions,
                resourceManager,
                usedFeatures,
                clipDistancesWritten,
                identification,
                layerInputAttr);
        }

        private ShaderProgram Generate(
            IReadOnlyList<Function> funcs,
            AttributeUsage attributeUsage,
            ShaderDefinitions definitions,
            ShaderDefinitions originalDefinitions,
            ResourceManager resourceManager,
            FeatureFlags usedFeatures,
            byte clipDistancesWritten,
            ShaderIdentification identification = ShaderIdentification.None,
            int layerInputAttr = 0)
        {
            var sInfo = StructuredProgram.MakeStructuredProgram(
                funcs,
                attributeUsage,
                definitions,
                resourceManager,
                Options.Flags.HasFlag(TranslationFlags.DebugMode));

            int geometryVerticesPerPrimitive = Definitions.OutputTopology switch
            {
                OutputTopology.LineStrip => 2,
                OutputTopology.TriangleStrip => 3,
                _ => 1
            };

            var info = new ShaderProgramInfo(
                resourceManager.GetConstantBufferDescriptors(),
                resourceManager.GetStorageBufferDescriptors(),
                resourceManager.GetTextureDescriptors(),
                resourceManager.GetImageDescriptors(),
                identification,
                layerInputAttr,
                originalDefinitions.Stage,
                geometryVerticesPerPrimitive,
                originalDefinitions.MaxOutputVertices,
                originalDefinitions.ThreadsPerInputPrimitive,
                usedFeatures.HasFlag(FeatureFlags.FragCoordXY),
                usedFeatures.HasFlag(FeatureFlags.InstanceId),
                usedFeatures.HasFlag(FeatureFlags.DrawParameters),
                usedFeatures.HasFlag(FeatureFlags.RtLayer),
                clipDistancesWritten,
                originalDefinitions.OmapTargets);

            var hostCapabilities = new HostCapabilities(
                GpuAccessor.QueryHostReducedPrecision(),
                GpuAccessor.QueryHostSupportsFragmentShaderInterlock(),
                GpuAccessor.QueryHostSupportsFragmentShaderOrderingIntel(),
                GpuAccessor.QueryHostSupportsGeometryShaderPassthrough(),
                GpuAccessor.QueryHostSupportsShaderBallot(),
                GpuAccessor.QueryHostSupportsShaderBarrierDivergence(),
                GpuAccessor.QueryHostSupportsTextureShadowLod(),
                GpuAccessor.QueryHostSupportsViewportMask());

            var parameters = new CodeGenParameters(attributeUsage, definitions, resourceManager.Properties, hostCapabilities, GpuAccessor, Options.TargetApi);

            return Options.TargetLanguage switch
            {
                TargetLanguage.Glsl => new ShaderProgram(info, TargetLanguage.Glsl, GlslGenerator.Generate(sInfo, parameters)),
                TargetLanguage.Spirv => new ShaderProgram(info, TargetLanguage.Spirv, SpirvGenerator.Generate(sInfo, parameters)),
                _ => throw new NotImplementedException(Options.TargetLanguage.ToString()),
            };
        }

        private ResourceManager CreateResourceManager(bool vertexAsCompute)
        {
            ResourceManager resourceManager = new(Definitions.Stage, GpuAccessor, GetResourceReservations());

            bool isTransformFeedbackEmulated = !GpuAccessor.QueryHostSupportsTransformFeedback() && GpuAccessor.QueryTransformFeedbackEnabled();

            if (isTransformFeedbackEmulated)
            {
                StructureType tfeDataStruct = new(new StructureField[]
                {
                    new StructureField(AggregateType.Array | AggregateType.U32, "data", 0)
                });

                for (int i = 0; i < ResourceReservations.TfeBuffersCount; i++)
                {
                    int binding = resourceManager.Reservations.GetTfeBufferStorageBufferBinding(i);
                    BufferDefinition tfeDataBuffer = new(BufferLayout.Std430, 1, binding, $"tfe_data{i}", tfeDataStruct);
                    resourceManager.Properties.AddOrUpdateStorageBuffer(tfeDataBuffer);
                }
            }

            if (vertexAsCompute)
            {
                StructureType vertexInfoStruct = new(new StructureField[]
                {
                    new StructureField(AggregateType.Vector4 | AggregateType.U32, "vertex_counts"),
                    new StructureField(AggregateType.Vector4 | AggregateType.U32, "geometry_counts"),
                    new StructureField(AggregateType.Array | AggregateType.Vector4 | AggregateType.U32, "vertex_strides", ResourceReservations.MaxVertexBufferTextures),
                    new StructureField(AggregateType.Array | AggregateType.Vector4 | AggregateType.U32, "vertex_offsets", ResourceReservations.MaxVertexBufferTextures),
                });

                int vertexInfoCbBinding = resourceManager.Reservations.GetVertexInfoConstantBufferBinding();
                BufferDefinition vertexInfoBuffer = new(BufferLayout.Std140, 0, vertexInfoCbBinding, "vb_info", vertexInfoStruct);
                resourceManager.Properties.AddOrUpdateConstantBuffer(vertexInfoBuffer);

                StructureType vertexOutputStruct = new(new StructureField[]
                {
                    new StructureField(AggregateType.Array | AggregateType.FP32, "data", 0)
                });

                int vertexOutputSbBinding = resourceManager.Reservations.GetVertexOutputStorageBufferBinding();
                BufferDefinition vertexOutputBuffer = new(BufferLayout.Std430, 1, vertexOutputSbBinding, "vertex_output", vertexOutputStruct);
                resourceManager.Properties.AddOrUpdateStorageBuffer(vertexOutputBuffer);

                if (Stage == ShaderStage.Vertex)
                {
                    int ibBinding = resourceManager.Reservations.GetIndexBufferTextureBinding();
                    TextureDefinition indexBuffer = new(2, ibBinding, "ib_data", SamplerType.TextureBuffer, TextureFormat.Unknown, TextureUsageFlags.None);
                    resourceManager.Properties.AddOrUpdateTexture(indexBuffer);

                    int inputMap = _program.AttributeUsage.UsedInputAttributes;

                    while (inputMap != 0)
                    {
                        int location = BitOperations.TrailingZeroCount(inputMap);
                        int binding = resourceManager.Reservations.GetVertexBufferTextureBinding(location);
                        TextureDefinition vaBuffer = new(2, binding, $"vb_data{location}", SamplerType.TextureBuffer, TextureFormat.Unknown, TextureUsageFlags.None);
                        resourceManager.Properties.AddOrUpdateTexture(vaBuffer);

                        inputMap &= ~(1 << location);
                    }
                }
                else if (Stage == ShaderStage.Geometry)
                {
                    int trbBinding = resourceManager.Reservations.GetTopologyRemapBufferTextureBinding();
                    TextureDefinition remapBuffer = new(2, trbBinding, "trb_data", SamplerType.TextureBuffer, TextureFormat.Unknown, TextureUsageFlags.None);
                    resourceManager.Properties.AddOrUpdateTexture(remapBuffer);

                    int geometryVbOutputSbBinding = resourceManager.Reservations.GetGeometryVertexOutputStorageBufferBinding();
                    BufferDefinition geometryVbOutputBuffer = new(BufferLayout.Std430, 1, geometryVbOutputSbBinding, "geometry_vb_output", vertexOutputStruct);
                    resourceManager.Properties.AddOrUpdateStorageBuffer(geometryVbOutputBuffer);

                    StructureType geometryIbOutputStruct = new(new StructureField[]
                    {
                        new StructureField(AggregateType.Array | AggregateType.U32, "data", 0)
                    });

                    int geometryIbOutputSbBinding = resourceManager.Reservations.GetGeometryIndexOutputStorageBufferBinding();
                    BufferDefinition geometryIbOutputBuffer = new(BufferLayout.Std430, 1, geometryIbOutputSbBinding, "geometry_ib_output", geometryIbOutputStruct);
                    resourceManager.Properties.AddOrUpdateStorageBuffer(geometryIbOutputBuffer);
                }

                resourceManager.SetVertexAsComputeLocalMemories(Definitions.Stage, Definitions.InputTopology);
            }

            return resourceManager;
        }

        private ShaderDefinitions GetDefinitions(bool vertexAsCompute)
        {
            if (vertexAsCompute)
            {
                return new ShaderDefinitions(ShaderStage.Compute, 32, 32, 1);
            }
            else
            {
                return Definitions;
            }
        }

        private FeatureFlags GetUsedFeatures(bool vertexAsCompute)
        {
            if (vertexAsCompute)
            {
                return UsedFeatures | FeatureFlags.VtgAsCompute;
            }
            else
            {
                return UsedFeatures;
            }
        }

        public ResourceReservations GetResourceReservations()
        {
            bool isTransformFeedbackEmulated = !GpuAccessor.QueryHostSupportsTransformFeedback() && GpuAccessor.QueryTransformFeedbackEnabled();

            IoUsage ioUsage = _program.GetIoUsage();

            if (Definitions.GpPassthrough)
            {
                ioUsage = ioUsage.Combine(_vertexOutput);
            }

            return new ResourceReservations(GpuAccessor, isTransformFeedbackEmulated, vertexAsCompute: true, _vertexOutput, ioUsage);
        }

        public void SetVertexOutputMapForGeometryAsCompute(TranslatorContext vertexContext)
        {
            _vertexOutput = vertexContext._program.GetIoUsage();
        }

        public ShaderProgram GenerateFeedbackForCompute()
        {
            var attributeUsage = new AttributeUsage(GpuAccessor);
            var resourceManager = new ResourceManager(ShaderStage.Vertex, GpuAccessor);

            var reservations = GetResourceReservations();

            int vertexInfoCbBinding = reservations.GetVertexInfoConstantBufferBinding();
            int vertexDataSbBinding = reservations.GetVertexOutputStorageBufferBinding();
            int indexDataTexBinding = reservations.GetIndexBufferTextureBinding();
            int remapDataTexBinding = reservations.GetTopologyRemapBufferTextureBinding();

            StructureType vertexInfoStruct = new(new StructureField[]
            {
                new StructureField(AggregateType.Vector4 | AggregateType.U32, "vertex_counts"),
                new StructureField(AggregateType.Vector4 | AggregateType.U32, "geometry_counts"),
            });

            BufferDefinition vertexInfoBuffer = new(BufferLayout.Std140, 0, vertexInfoCbBinding, "vb_info", vertexInfoStruct);
            resourceManager.Properties.AddOrUpdateConstantBuffer(vertexInfoBuffer);

            StructureType vertexInputStruct = new(new StructureField[]
            {
                new StructureField(AggregateType.Array | AggregateType.FP32, "data", 0)
            });

            BufferDefinition vertexOutputBuffer = new(BufferLayout.Std430, 1, vertexDataSbBinding, "vb_input", vertexInputStruct);
            resourceManager.Properties.AddOrUpdateStorageBuffer(vertexOutputBuffer);

            int baseIndexMemoryId = resourceManager.Properties.AddLocalMemory(new("base_index", AggregateType.U32));
            int indexCountMemoryId = resourceManager.Properties.AddLocalMemory(new("index_count", AggregateType.U32));
            int currentVertexMemoryId = resourceManager.Properties.AddLocalMemory(new("current_vertex", AggregateType.U32));
            int outputVertexMemoryId = resourceManager.Properties.AddLocalMemory(new("output_vertex", AggregateType.U32));

            TextureDefinition indexBuffer = new(2, indexDataTexBinding, "ib_data", SamplerType.TextureBuffer, TextureFormat.Unknown, TextureUsageFlags.None);
            resourceManager.Properties.AddOrUpdateTexture(indexBuffer);

            TextureDefinition remapBuffer = new(2, remapDataTexBinding, "trb_data", SamplerType.TextureBuffer, TextureFormat.Unknown, TextureUsageFlags.None);
            resourceManager.Properties.AddOrUpdateTexture(remapBuffer);

            StructureType tfeDataStruct = new(new StructureField[]
            {
                new StructureField(AggregateType.Array | AggregateType.U32, "data", 0)
            });

            for (int i = 0; i < ResourceReservations.TfeBuffersCount; i++)
            {
                int binding = reservations.GetTfeBufferStorageBufferBinding(i);
                BufferDefinition tfeDataBuffer = new(BufferLayout.Std430, 1, binding, $"tfe_data{i}", tfeDataStruct);
                resourceManager.Properties.AddOrUpdateStorageBuffer(tfeDataBuffer);
            }

            var context = new EmitterContext();

            // Main loop start.

            Operand lblLoopHead = Label();

            Operand ibIndexCount = context.Load(StorageKind.ConstantBuffer, vertexInfoCbBinding, Const(1), Const(1));
            Operand primitiveRestartValue = context.Load(StorageKind.ConstantBuffer, vertexInfoCbBinding, Const(1), Const(2));

            context.Store(StorageKind.LocalMemory, outputVertexMemoryId, Const(0));

            context.MarkLabel(lblLoopHead);

            Operand baseIndex = context.Load(StorageKind.LocalMemory, baseIndexMemoryId);

            context.Store(StorageKind.LocalMemory, indexCountMemoryId, Const(0));

            // First inner loop:
            // Find primitive restart value on the index buffer (if any), count amount of vertices until the restart or end of the buffer.

            Operand lblCounterLoopHead = Label();
            Operand lblCounterLoopExit = Label();

            context.MarkLabel(lblCounterLoopHead);

            Operand indexCount = context.Load(StorageKind.LocalMemory, indexCountMemoryId);
            Operand currentIndexValue = Local();

            Operand currentIndex = context.IAdd(baseIndex, indexCount);

            context.TextureSample(
                SamplerType.TextureBuffer,
                TextureFlags.IntCoords,
                indexDataTexBinding,
                1,
                new[] { currentIndexValue },
                new[] { currentIndex });

            context.Store(StorageKind.LocalMemory, indexCountMemoryId, context.IAdd(indexCount, Const(1)));
            context.BranchIfTrue(lblCounterLoopExit, context.ICompareEqual(currentIndexValue, primitiveRestartValue));
            context.BranchIfTrue(lblCounterLoopHead, context.ICompareLess(currentIndex, ibIndexCount));

            context.MarkLabel(lblCounterLoopExit);

            // Calculate non-strip vertex count based on contiguous indices that we counted.

            indexCount = context.Load(StorageKind.LocalMemory, indexCountMemoryId);

            Operand vertexStripCount = context.ISubtract(indexCount, Const(1));
            Operand vertexCount;

            if (Definitions.Stage == ShaderStage.Geometry)
            {
                vertexCount = Definitions.OutputTopology switch
                {
                    OutputTopology.LineStrip => context.ShiftLeft(context.ISubtract(vertexStripCount, Const(1)), Const(1)),
                    OutputTopology.TriangleStrip => context.IMultiply(context.ISubtract(vertexStripCount, Const(2)), Const(3)),
                    _ => vertexStripCount,
                };
            }
            else
            {
                // Note that this will feedback incomplete primitives, if they exist on the input, in some cases.
                // It might be worth "rounding" the vertex count in all cases in the future, but it might
                // require doing an integer division on the shader.

                Operand SelectIfGreaterThanOne(Operand value)
                {
                    return context.ConditionalSelect(context.ICompareGreater(value, Const(1)), value, Const(0));
                }

                vertexCount = Definitions.InputTopologyForVertex switch
                {
                    InputTopologyForVertex.LineLoop => context.ShiftLeft(SelectIfGreaterThanOne(vertexStripCount), Const(1)),
                    InputTopologyForVertex.LineStrip => context.ShiftLeft(context.ISubtract(vertexStripCount, Const(1)), Const(1)),
                    InputTopologyForVertex.LineStripAdjacency => context.ShiftLeft(context.ISubtract(vertexStripCount, Const(3)), Const(2)),
                    InputTopologyForVertex.TriangleStrip or
                    InputTopologyForVertex.TriangleFan or
                    InputTopologyForVertex.Polygon or
                    InputTopologyForVertex.LineStripAdjacency => context.IMultiply(context.ISubtract(vertexStripCount, Const(2)), Const(3)),
                    InputTopologyForVertex.Quads => context.IMultiply(context.ShiftRightS32(vertexStripCount, Const(2)), Const(6)), // In triangles.
                    InputTopologyForVertex.QuadStrip => context.IMultiply(context.ShiftRightS32(context.ISubtract(vertexStripCount, Const(2)), Const(1)), Const(6)), // In triangles.
                    _ => vertexStripCount,
                };
            }

            Operand lblLoopEnd = Label();
            context.BranchIfTrue(lblLoopEnd, context.ICompareLessOrEqual(vertexCount, Const(0)));

            // Second inner loop:
            // Write output vertices for a batch of primitives into the transform feedback buffer.

            context.Store(StorageKind.LocalMemory, currentVertexMemoryId, Const(0));

            Operand lblFeedbackLoopHead = Label();

            context.MarkLabel(lblFeedbackLoopHead);

            Operand currentVertex = context.Load(StorageKind.LocalMemory, currentVertexMemoryId);

            Operand vertexRemapIndex = Local();

            context.TextureSample(
                SamplerType.TextureBuffer,
                TextureFlags.IntCoords,
                remapDataTexBinding,
                1,
                new[] { vertexRemapIndex },
                new[] { currentVertex });

            Operand vertexIndex = Local();

            context.TextureSample(
                SamplerType.TextureBuffer,
                TextureFlags.IntCoords,
                indexDataTexBinding,
                1,
                new[] { vertexIndex },
                new[] { context.IAdd(baseIndex, vertexRemapIndex) });

            Operand outputVertex = context.Load(StorageKind.LocalMemory, outputVertexMemoryId);

            for (int tfbIndex = 0; tfbIndex < ResourceReservations.TfeBuffersCount; tfbIndex++)
            {
                var locations = GpuAccessor.QueryTransformFeedbackVaryingLocations(tfbIndex);
                var stride = GpuAccessor.QueryTransformFeedbackStride(tfbIndex);

                Operand outputBaseOffset = context.Load(StorageKind.ConstantBuffer, SupportBuffer.Binding, Const((int)SupportBufferField.TfeOffset), Const(tfbIndex));
                Operand outputSize = context.Load(StorageKind.ConstantBuffer, SupportBuffer.Binding, Const((int)SupportBufferField.TfeSize), Const(tfbIndex));
                Operand instanceOffset = context.Load(StorageKind.Input, IoVariable.GlobalInvocationId, null, Const(0));
                Operand verticesPerInstance = context.Load(StorageKind.ConstantBuffer, SupportBuffer.Binding, Const((int)SupportBufferField.TfeVertexCount));
                Operand baseVertex = context.IMultiply(instanceOffset, verticesPerInstance);
                Operand outputVertexOffset = context.IMultiply(context.IAdd(baseVertex, outputVertex), Const(stride / 4));
                outputBaseOffset = context.IAdd(outputBaseOffset, outputVertexOffset);

                Operand inputBaseOffset = context.IMultiply(context.IAdd(baseVertex, vertexIndex), Const(reservations.OutputSizePerInvocation));

                for (int j = 0; j < locations.Length; j++)
                {
                    byte location = locations[j];
                    if (location == 0xff)
                    {
                        continue;
                    }

                    if (!Instructions.AttributeMap.TryGetIoDefinition(
                        Definitions.Stage,
                        location * 4,
                        out IoVariable ioVariable,
                        out int attrLocation,
                        out int attrComponent))
                    {
                        continue;
                    }

                    if (!reservations.TryGetOffset(StorageKind.Output, ioVariable, attrLocation, attrComponent, out int inputOffset))
                    {
                        continue;
                    }

                    Operand outputOffset = context.IAdd(outputBaseOffset, Const(j));

                    Operand lblOutOfRange = Label();
                    context.BranchIfFalse(lblOutOfRange, context.ICompareLess(outputOffset, outputSize));

                    Operand vertexOffset = inputOffset != 0 ? context.IAdd(inputBaseOffset, Const(inputOffset)) : inputBaseOffset;
                    Operand value = context.Load(StorageKind.StorageBuffer, vertexDataSbBinding, Const(0), vertexOffset);

                    int binding = reservations.GetTfeBufferStorageBufferBinding(tfbIndex);

                    context.Store(StorageKind.StorageBuffer, binding, Const(0), outputOffset, value);

                    context.MarkLabel(lblOutOfRange);
                }
            }

            currentVertex = context.IAdd(currentVertex, Const(1));
            context.Store(StorageKind.LocalMemory, currentVertexMemoryId, currentVertex);
            context.Store(StorageKind.LocalMemory, outputVertexMemoryId, context.IAdd(outputVertex, Const(1)));
            context.BranchIfTrue(lblFeedbackLoopHead, context.ICompareLess(currentVertex, vertexCount));

            // End of the main loop, continue until we reach the end of the index buffer.

            context.MarkLabel(lblLoopEnd);

            Operand newBaseIndex = context.IAdd(baseIndex, indexCount);

            context.Store(StorageKind.LocalMemory, baseIndexMemoryId, newBaseIndex);
            context.BranchIfTrue(lblLoopHead, context.ICompareLess(newBaseIndex, ibIndexCount));

            var operations = context.GetOperations();
            var cfg = ControlFlowGraph.Create(operations);
            var function = new Function(cfg.Blocks, "main", false, 0, 0);

            var definitions = new ShaderDefinitions(ShaderStage.Compute, 1, 1, 1);

            return Generate(
                new[] { function },
                attributeUsage,
                definitions,
                definitions,
                resourceManager,
                FeatureFlags.None,
                0);
        }

        public ShaderProgram GenerateVertexPassthroughForCompute()
        {
            var attributeUsage = new AttributeUsage(GpuAccessor);
            var resourceManager = new ResourceManager(ShaderStage.Vertex, GpuAccessor);

            var reservations = GetResourceReservations();

            int vertexInfoCbBinding = reservations.GetVertexInfoConstantBufferBinding();

            if (Stage == ShaderStage.Vertex)
            {
                StructureType vertexInfoStruct = new(new StructureField[]
                {
                    new StructureField(AggregateType.Vector4 | AggregateType.U32, "vertex_counts"),
                });

                BufferDefinition vertexInfoBuffer = new(BufferLayout.Std140, 0, vertexInfoCbBinding, "vb_info", vertexInfoStruct);
                resourceManager.Properties.AddOrUpdateConstantBuffer(vertexInfoBuffer);
            }

            StructureType vertexInputStruct = new(new StructureField[]
            {
                new StructureField(AggregateType.Array | AggregateType.FP32, "data", 0)
            });

            int vertexDataSbBinding = reservations.GetVertexOutputStorageBufferBinding();
            BufferDefinition vertexOutputBuffer = new(BufferLayout.Std430, 1, vertexDataSbBinding, "vb_input", vertexInputStruct);
            resourceManager.Properties.AddOrUpdateStorageBuffer(vertexOutputBuffer);

            var context = new EmitterContext();

            Operand vertexIndex = Options.TargetApi == TargetApi.OpenGL
                ? context.Load(StorageKind.Input, IoVariable.VertexId)
                : context.Load(StorageKind.Input, IoVariable.VertexIndex);

            if (Stage == ShaderStage.Vertex)
            {
                Operand vertexCount = context.Load(StorageKind.ConstantBuffer, vertexInfoCbBinding, Const(0), Const(0));

                // Base instance will be always zero when this shader is used, so which one we use here doesn't really matter.
                Operand instanceId = Options.TargetApi == TargetApi.OpenGL
                    ? context.Load(StorageKind.Input, IoVariable.InstanceId)
                    : context.Load(StorageKind.Input, IoVariable.InstanceIndex);

                vertexIndex = context.IAdd(context.IMultiply(instanceId, vertexCount), vertexIndex);
            }

            Operand baseOffset = context.IMultiply(vertexIndex, Const(reservations.OutputSizePerInvocation));

            foreach ((IoDefinition ioDefinition, int inputOffset) in reservations.Offsets)
            {
                if (ioDefinition.StorageKind != StorageKind.Output)
                {
                    continue;
                }

                Operand vertexOffset = inputOffset != 0 ? context.IAdd(baseOffset, Const(inputOffset)) : baseOffset;
                Operand value = context.Load(StorageKind.StorageBuffer, vertexDataSbBinding, Const(0), vertexOffset);

                if (ioDefinition.IoVariable == IoVariable.UserDefined)
                {
                    context.Store(StorageKind.Output, ioDefinition.IoVariable, null, Const(ioDefinition.Location), Const(ioDefinition.Component), value);
                    attributeUsage.SetOutputUserAttribute(ioDefinition.Location);
                }
                else if (ResourceReservations.IsVectorOrArrayVariable(ioDefinition.IoVariable))
                {
                    context.Store(StorageKind.Output, ioDefinition.IoVariable, null, Const(ioDefinition.Component), value);
                }
                else
                {
                    context.Store(StorageKind.Output, ioDefinition.IoVariable, null, value);
                }
            }

            var operations = context.GetOperations();
            var cfg = ControlFlowGraph.Create(operations);
            var function = new Function(cfg.Blocks, "main", false, 0, 0);

            var transformFeedbackOutputs = GetTransformFeedbackOutputs(GpuAccessor, out ulong transformFeedbackVecMap);

            var definitions = new ShaderDefinitions(ShaderStage.Vertex, transformFeedbackVecMap, transformFeedbackOutputs)
            {
                LastInVertexPipeline = true
            };

            return Generate(
                new[] { function },
                attributeUsage,
                definitions,
                definitions,
                resourceManager,
                FeatureFlags.None,
                0);
        }

        public ShaderProgram GenerateGeometryPassthrough()
        {
            int outputAttributesMask = AttributeUsage.UsedOutputAttributes;
            int layerOutputAttr = LayerOutputAttribute;

            OutputTopology outputTopology;
            int maxOutputVertices;

            switch (Definitions.InputTopology)
            {
                case InputTopology.Points:
                    outputTopology = OutputTopology.PointList;
                    maxOutputVertices = 1;
                    break;
                case InputTopology.Lines:
                case InputTopology.LinesAdjacency:
                    outputTopology = OutputTopology.LineStrip;
                    maxOutputVertices = 2;
                    break;
                default:
                    outputTopology = OutputTopology.TriangleStrip;
                    maxOutputVertices = 3;
                    break;
            }

            var attributeUsage = new AttributeUsage(GpuAccessor);
            var resourceManager = new ResourceManager(ShaderStage.Geometry, GpuAccessor);

            var context = new EmitterContext();

            for (int v = 0; v < maxOutputVertices; v++)
            {
                int outAttrsMask = outputAttributesMask;

                while (outAttrsMask != 0)
                {
                    int attrIndex = BitOperations.TrailingZeroCount(outAttrsMask);

                    outAttrsMask &= ~(1 << attrIndex);

                    for (int c = 0; c < 4; c++)
                    {
                        int attr = AttributeConsts.UserAttributeBase + attrIndex * 16 + c * 4;

                        Operand value = context.Load(StorageKind.Input, IoVariable.UserDefined, Const(attrIndex), Const(v), Const(c));

                        if (attr == layerOutputAttr)
                        {
                            context.Store(StorageKind.Output, IoVariable.Layer, null, value);
                        }
                        else
                        {
                            context.Store(StorageKind.Output, IoVariable.UserDefined, null, Const(attrIndex), Const(c), value);
                        }
                    }
                }

                for (int c = 0; c < 4; c++)
                {
                    Operand value = context.Load(StorageKind.Input, IoVariable.Position, Const(v), Const(c));

                    context.Store(StorageKind.Output, IoVariable.Position, null, Const(c), value);
                }

                context.EmitVertex();
            }

            context.EndPrimitive();

            var operations = context.GetOperations();
            var cfg = ControlFlowGraph.Create(operations);
            var function = new Function(cfg.Blocks, "main", false, 0, 0);

            var definitions = new ShaderDefinitions(
                ShaderStage.Geometry,
                GpuAccessor.QueryGraphicsState(),
                false,
                1,
                outputTopology,
                maxOutputVertices);

            return Generate(
                new[] { function },
                attributeUsage,
                definitions,
                definitions,
                resourceManager,
                FeatureFlags.RtLayer,
                0);
        }
    }
}
