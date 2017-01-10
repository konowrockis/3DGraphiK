using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media.Media3D;

namespace GraphiK3D.Rendering.VertexShading
{
    struct VertexShaderIn
    {
        public Point3D Pos;
        public Vector3D Normal;
        public Vector3D Color;
    }
}
