using Ryujinx.Graphics.Shader.IntermediateRepresentation;
using Ryujinx.Graphics.Shader.Translation.Optimizations;
using System.Collections.Generic;

using static Ryujinx.Graphics.Shader.IntermediateRepresentation.OperandHelper;

namespace Ryujinx.Graphics.Shader.Translation.Transforms
{
    class VertexToCompute : ITransformPass
    {
        public static bool IsEnabled(IGpuAccessor gpuAccessor, ShaderStage stage, TargetLanguage targetLanguage, FeatureFlags usedFeatures)
        {
            return usedFeatures.HasFlag(FeatureFlags.VtgAsCompute);
        }

        public static LinkedListNode<INode> RunPass(TransformContext context, LinkedListNode<INode> node)
        {
            Operation operation = (Operation)node.Value;

            LinkedListNode<INode> newNode = node;

            if (operation.Inst == Instruction.Load && operation.StorageKind == StorageKind.Input)
            {
                Operand dest = operation.Dest;

                switch ((IoVariable)operation.GetSource(0).Value)
                {
                    case IoVariable.BaseInstance:
                        newNode = GenerateBaseInstanceLoad(context.ResourceManager, node, dest);
                        break;
                    case IoVariable.BaseVertex:
                        newNode = GenerateBaseVertexLoad(context.ResourceManager, node, dest);
                        break;
                    case IoVariable.InstanceId:
                        newNode = GenerateInstanceIdLoad(node, dest);
                        break;
                    case IoVariable.InstanceIndex:
                        newNode = GenerateInstanceIndexLoad(context.ResourceManager, node, dest);
                        break;
                    case IoVariable.VertexId:
                    case IoVariable.VertexIndex:
                        newNode = GenerateVertexIndexLoad(context.ResourceManager, node, dest);
                        break;
                    case IoVariable.UserDefined:
                        int location = operation.GetSource(1).Value;
                        int component = operation.GetSource(2).Value;

                        Operand vertexElemOffset = GenerateVertexOffset(context.ResourceManager, node, location, component);

                        Operand temp = component > 0 ? Local() : dest;

                        newNode = node.List.AddBefore(node, new TextureOperation(
                            Instruction.TextureSample,
                            SamplerType.TextureBuffer,
                            TextureFormat.Unknown,
                            TextureFlags.IntCoords,
                            context.ResourceManager.Reservations.GetVertexBufferTextureBinding(location),
                            1,
                            new[] { temp },
                            new[] { vertexElemOffset }));

                        if (component > 0)
                        {
                            newNode = CopyMasked(context.ResourceManager, newNode, location, component, dest, temp);
                        }
                        break;
                    case IoVariable.GlobalInvocationId:
                        // We generate that for those compute shaders.
                        break;
                    default:
                        context.GpuAccessor.Log($"Invalid input \"{(IoVariable)operation.GetSource(0).Value}\".");
                        break;
                }
            }
            else if (operation.Inst == Instruction.Load && operation.StorageKind == StorageKind.Output)
            {
                if (TryGetOutputOffset(context.ResourceManager, operation, out int outputOffset))
                {
                    newNode = node.List.AddBefore(node, new Operation(
                        Instruction.Load,
                        StorageKind.LocalMemory,
                        operation.Dest,
                        new[] { Const(context.ResourceManager.LocalVertexDataMemoryId), Const(outputOffset) }));
                }
                else
                {
                    context.GpuAccessor.Log($"Invalid output \"{(IoVariable)operation.GetSource(0).Value}\".");
                }
            }
            else if (operation.Inst == Instruction.Store && operation.StorageKind == StorageKind.Output)
            {
                if (TryGetOutputOffset(context.ResourceManager, operation, out int outputOffset))
                {
                    Operand value = operation.GetSource(operation.SourcesCount - 1);

                    newNode = node.List.AddBefore(node, new Operation(
                        Instruction.Store,
                        StorageKind.LocalMemory,
                        (Operand)null,
                        new[] { Const(context.ResourceManager.LocalVertexDataMemoryId), Const(outputOffset), value }));
                }
                else
                {
                    context.GpuAccessor.Log($"Invalid output \"{(IoVariable)operation.GetSource(0).Value}\".");
                }
            }

            if (newNode != node)
            {
                Utils.DeleteNode(node, operation);
            }

            return newNode;
        }

        private static Operand GenerateVertexOffset(ResourceManager resourceManager, LinkedListNode<INode> node, int location, int component)
        {
            Operand vertexId = Local();
            GenerateVertexIdLoad(resourceManager, node, vertexId);

            Operand vertexStride = Local();
            int vertexInfoCbBinding = resourceManager.Reservations.GetVertexInfoConstantBufferBinding();
            node.List.AddBefore(node, new Operation(
                Instruction.Load,
                StorageKind.ConstantBuffer,
                vertexStride,
                new[] { Const(vertexInfoCbBinding), Const(1), Const(location), Const(0) }));

            Operand vertexBaseOffset = Local();
            node.List.AddBefore(node, new Operation(Instruction.Multiply, vertexBaseOffset, new[] { vertexId, vertexStride }));

            Operand vertexElemOffset;

            if (component != 0)
            {
                vertexElemOffset = Local();

                node.List.AddBefore(node, new Operation(Instruction.Add, vertexElemOffset, new[] { vertexBaseOffset, Const(component) }));
            }
            else
            {
                vertexElemOffset = vertexBaseOffset;
            }

            return vertexElemOffset;
        }

        private static LinkedListNode<INode> CopyMasked(
            ResourceManager resourceManager,
            LinkedListNode<INode> node,
            int location,
            int component,
            Operand dest,
            Operand src)
        {
            Operand componentExists = Local();
            int vertexInfoCbBinding = resourceManager.Reservations.GetVertexInfoConstantBufferBinding();
            node = node.List.AddAfter(node, new Operation(
                Instruction.Load,
                StorageKind.ConstantBuffer,
                componentExists,
                new[] { Const(vertexInfoCbBinding), Const(1), Const(location), Const(component) }));

            return node.List.AddAfter(node, new Operation(
                Instruction.ConditionalSelect,
                dest,
                new[] { componentExists, src, ConstF(0) }));
        }

        private static LinkedListNode<INode> GenerateBaseVertexLoad(ResourceManager resourceManager, LinkedListNode<INode> node, Operand dest)
        {
            int vertexInfoCbBinding = resourceManager.Reservations.GetVertexInfoConstantBufferBinding();

            return node.List.AddBefore(node, new Operation(
                Instruction.Load,
                StorageKind.ConstantBuffer,
                dest,
                new[] { Const(vertexInfoCbBinding), Const(0), Const(2) }));
        }

        private static LinkedListNode<INode> GenerateBaseInstanceLoad(ResourceManager resourceManager, LinkedListNode<INode> node, Operand dest)
        {
            int vertexInfoCbBinding = resourceManager.Reservations.GetVertexInfoConstantBufferBinding();

            return node.List.AddBefore(node, new Operation(
                Instruction.Load,
                StorageKind.ConstantBuffer,
                dest,
                new[] { Const(vertexInfoCbBinding), Const(0), Const(3) }));
        }

        private static LinkedListNode<INode> GenerateVertexIndexLoad(ResourceManager resourceManager, LinkedListNode<INode> node, Operand dest)
        {
            Operand baseVertex = Local();
            Operand vertexId = Local();

            GenerateBaseVertexLoad(resourceManager, node, baseVertex);
            GenerateVertexIdLoad(resourceManager, node, vertexId);

            return node.List.AddBefore(node, new Operation( Instruction.Add, dest, new[] { baseVertex, vertexId }));
        }

        private static LinkedListNode<INode> GenerateInstanceIndexLoad(ResourceManager resourceManager, LinkedListNode<INode> node, Operand dest)
        {
            Operand baseInstance = Local();
            Operand instanceId = Local();

            GenerateBaseInstanceLoad(resourceManager, node, baseInstance);

            node.List.AddBefore(node, new Operation(
                Instruction.Load,
                StorageKind.Input,
                instanceId,
                new[] { Const((int)IoVariable.GlobalInvocationId), Const(1) }));

            return node.List.AddBefore(node, new Operation( Instruction.Add, dest, new[] { baseInstance, instanceId }));
        }

        private static LinkedListNode<INode> GenerateVertexIdLoad(ResourceManager resourceManager, LinkedListNode<INode> node, Operand dest)
        {
            Operand[] sources = new Operand[] { Const(resourceManager.LocalVertexIndexMemoryId) };

            return node.List.AddBefore(node, new Operation(Instruction.Load, StorageKind.LocalMemory, dest, sources));
        }

        private static LinkedListNode<INode> GenerateInstanceIdLoad(LinkedListNode<INode> node, Operand dest)
        {
            Operand[] sources = new Operand[] { Const((int)IoVariable.GlobalInvocationId), Const(1) };

            return node.List.AddBefore(node, new Operation(Instruction.Load, StorageKind.Input, dest, sources));
        }

        private static bool TryGetOutputOffset(ResourceManager resourceManager, Operation operation, out int outputOffset)
        {
            bool isStore = operation.Inst == Instruction.Store;

            IoVariable ioVariable = (IoVariable)operation.GetSource(0).Value;

            bool isValidOutput;

            if (ioVariable == IoVariable.UserDefined)
            {
                int location = operation.GetSource(1).Value;
                int component = operation.GetSource(2).Value;

                isValidOutput = resourceManager.Reservations.TryGetOutputOffset(location, component, out outputOffset);
            }
            else
            {
                if (operation.SourcesCount > (isStore ? 2 : 1))
                {
                    int component = operation.GetSource(1).Value;

                    isValidOutput = resourceManager.Reservations.TryGetOutputOffset(ioVariable, component, out outputOffset);
                }
                else
                {
                    isValidOutput = resourceManager.Reservations.TryGetOutputOffset(ioVariable, out outputOffset);
                }
            }

            return isValidOutput;
        }
    }
}