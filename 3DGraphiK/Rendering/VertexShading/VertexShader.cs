using System.Windows.Media.Media3D;

namespace GraphiK3D.Rendering.VertexShading
{
    static class VertexShader
    {
        public static void Apply(VertexShaderIn[] vertexIn, VertexShaderOut[] vertexOut, Matrix3D transformation)
        {
            for (int i = 0; i < vertexIn.Length; i++)
            {
                vertexOut[i].Color = vertexIn[i].Color;
                vertexOut[i].ModelPos = vertexIn[i].Pos;
                vertexOut[i].Normal = vertexIn[i].Normal;

                vertexOut[i].Pos = transformation.Transform(vertexIn[i].Pos);
            }
        }
    }
}
