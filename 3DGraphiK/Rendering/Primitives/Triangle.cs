using System.Windows.Media.Media3D;
using GraphiK3D.Rendering.VertexShading;

namespace GraphiK3D.Rendering.Primitives
{
    struct Triangle
    {
        public VertexShaderOut v1;
        public VertexShaderOut v2;
        public VertexShaderOut v3;

        public Vector3D Normal;
         
        public bool Visible;
    }
}
