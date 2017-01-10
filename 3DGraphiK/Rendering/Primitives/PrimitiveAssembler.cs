using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media.Media3D;
using GraphiK3D.Rendering.VertexShading;

namespace GraphiK3D.Rendering.Primitives
{
    static class PrimitivesAssembler
    {
        public static void Assemble(VertexShaderOut[] vertexBuffer, int[] indices, Triangle[] primitivesBuffer, Vector3D[] normals)
        {
            for (int i = 0; i < indices.Length; i += 3)
            {
                int j = i / 3;

                primitivesBuffer[j].v1 = vertexBuffer[indices[i]];
                primitivesBuffer[j].v2 = vertexBuffer[indices[i + 1]];
                primitivesBuffer[j].v3 = vertexBuffer[indices[i + 2]];
                primitivesBuffer[j].Normal = normals[j];
                primitivesBuffer[j].Visible = true;
            }
        }
    }
}
