using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media.Media3D;
using GraphiK3D.Rendering.Primitives;

namespace GraphiK3D.Rendering.Culling
{
    static class BackFaceCulling
    {
        public static void Cull(Triangle[] primitivesBuffer, Point3D camera)
        {
            for (int i = 0; i < primitivesBuffer.Length; i++)
            {
                if (Vector3D.DotProduct(primitivesBuffer[i].v1.ModelPos - camera, primitivesBuffer[i].Normal) > 0)
                {
                    primitivesBuffer[i].Visible = false;
                }
            }
        }
    }
}
