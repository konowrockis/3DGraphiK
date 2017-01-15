using System;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Media.Media3D;
using GraphiK3D.Models;

namespace GraphiK3D.Rendering
{
    unsafe class Renderer
    {
        public struct float3
        {
            public float x;
            public float y;
            public float z;

            public float3(Point3D v)
            {
                x = (float)v.X;
                y = (float)v.Y;
                z = (float)v.Z;
            }

            public float3(Vector3D v)
            {
                x = (float)v.X;
                y = (float)v.Y;
                z = (float)v.Z;
            }
        }

        struct float4
        {
            public float x;
            public float y;
            public float z;
            public float w;
        }

        [DllImport("3DGraphiK.CudaRasterizer.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void Init(float3[] vertices, float3[] normals, float3[] colors, int[] indices, int numberOfVertices, int numberOfFaces);

        [DllImport("3DGraphiK.CudaRasterizer.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void SetTransformation(float[] transformation, float3 camera);

        [DllImport("3DGraphiK.CudaRasterizer.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void Rasterize();

        [DllImport("3DGraphiK.CudaRasterizer.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void Resize(int w, int h, IntPtr backBuffer);


        //private Rasterizer rasterizer;
        private Model model;

        private Matrix3D projectionMatrix;

        private bool isInitialized = false;

        public Renderer()
        {
            Init();   
        }

        public void CreateRasterizer(int width, int height)
        {
            //Resize(width, height, bmp.BackBuffer);

            isInitialized = true;
        }
        
        private Matrix3D GetProjectionMatrix(double fov = 60, double nearPlane = 0.1, double farPlane = 100)
        {
            double scale = Math.Tan(fov * 0.5 * Math.PI / 180) * nearPlane;
            double r = scale * 16 / 9;
            double t = scale;

            return new Matrix3D(
                nearPlane / r, 0, 0, 0,
                0, nearPlane / t, 0, 0,
                0, 0, -(farPlane + nearPlane) / (farPlane - nearPlane), -1,
                0, 0, -2 * farPlane * nearPlane / (farPlane - nearPlane), 0);
        }

        Point3D Camera = new Point3D(0, 1, -6);
        public Quaternion rotation = new Quaternion();

        public void Init()
        {
            model = ModelLoader.Load(@"Assets\teapot.off");
            projectionMatrix = GetProjectionMatrix();

            Init(
                model.Vertices.Select(v => new float3(v)).ToArray(), 
                model.Normals.Select(v => new float3(v)).ToArray(), 
                model.Colors.Select(v => new float3(v)).ToArray(), 
                model.Indices, model.Vertices.Length, model.NumberOfTriangles
            );
        }

        public void Redraw()
        {
            if (!isInitialized)
            {
                return;
            }

            Matrix3D transformation = Matrix3D.Identity;
            transformation.Rotate(new Quaternion(rotation.Axis, -rotation.Angle));
            Point3D camera = transformation.Transform(Camera);
            //rasterizer.light = transformation.Transform(new Point3D(10, 10, -3));

            transformation = Matrix3D.Identity;
            transformation.Rotate(rotation);
            transformation.Translate(-(Vector3D)Camera);
            transformation.Append(projectionMatrix);
            transformation.Translate(new Vector3D(1, 1, 0));
            //transformation.Scale(new Vector3D(bmp.Width / 2, bmp.Height / 2, 1));

            SetTransformation(new float[] {
                (float)transformation.M11, (float)transformation.M21, (float)transformation.M31, (float)transformation.OffsetX,
                (float)transformation.M12, (float)transformation.M22, (float)transformation.M32, (float)transformation.OffsetY,
                (float)transformation.M13, (float)transformation.M23, (float)transformation.M33, (float)transformation.OffsetZ,
                (float)transformation.M14, (float)transformation.M24, (float)transformation.M34, (float)transformation.M44
            }, new float3(camera));

            Rasterize();
        }
    }
}

