using System;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Media.Media3D;
using GraphiK3D.Models;
using GraphiK3D.Rendering.Culling;
using GraphiK3D.Rendering.Primitives;
using GraphiK3D.Rendering.VertexShading;

namespace GraphiK3D.Rendering
{
    unsafe class Renderer
    {
        struct float3
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
        private static extern float Rasterize();

        [DllImport("3DGraphiK.CudaRasterizer.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void Resize(int w, int h, IntPtr backBuffer);


        //private Rasterizer rasterizer;
        private Model model;

        private Matrix3D projectionMatrix;

        private VertexShaderIn[] vertexBufferIn;
        private VertexShaderOut[] vertexBufferOut;
        private Triangle[] primitivesBuffer;

        private bool isInitialized = false;

        public Renderer()
        {
            Init();   
        }

        private WriteableBitmap bmp;

        public ImageSource CreateRasterizer(int width, int height)
        {
            /*if (rasterizer == null)
            {
                rasterizer = new Rasterizer(width, height);
            }
            else
            {
                rasterizer.SetCanvasSize(width, height);
            }

            return rasterizer.GetSource();
            
             */

            bmp = new WriteableBitmap(width, height, 96, 96, PixelFormats.Bgr32, null);

            Resize(width, height, bmp.BackBuffer);

            isInitialized = true;

            return bmp;
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

            vertexBufferIn = new VertexShaderIn[model.Vertices.Length];
            vertexBufferOut = new VertexShaderOut[model.Vertices.Length];

            for (int i = 0; i < model.Vertices.Length; i++)
            {
                vertexBufferIn[i].Pos = model.Vertices[i];
                vertexBufferIn[i].Normal = model.NormalVertices[i];
                vertexBufferIn[i].Color = model.Colors[i];

                vertexBufferOut[i] = new VertexShaderOut();
            }

            primitivesBuffer = new Triangle[model.NumberOfTriangles];

            Init(
                model.Vertices.Select(v => new float3(v)).ToArray(), 
                model.Normals.Select(v => new float3(v)).ToArray(), 
                model.Colors.Select(v => new float3(v)).ToArray(), 
                model.Indices, model.Vertices.Length, model.NumberOfTriangles
            );
        }

        //public void Redraw()
        //{
        //    if (rasterizer == null)
        //    {
        //        return;
        //    }

        //    rasterizer.Begin();
        //    rasterizer.Clear(0x000000);

        //    Matrix3D transformation = Matrix3D.Identity;
        //    transformation.Rotate(new Quaternion(rotation.Axis, -rotation.Angle));
        //    Point3D camera = transformation.Transform(Camera);
        //    rasterizer.light = transformation.Transform(new Point3D(10, 10, -3));

        //    transformation = Matrix3D.Identity;
        //    transformation.Rotate(rotation);
        //    transformation.Translate(-(Vector3D)Camera);
        //    transformation.Append(projectionMatrix);

        //    SetTransformation(new float[] {
        //        (float)transformation.M11, (float)transformation.M21, (float)transformation.M31, (float)transformation.OffsetX,
        //        (float)transformation.M12, (float)transformation.M22, (float)transformation.M32, (float)transformation.OffsetY,
        //        (float)transformation.M13, (float)transformation.M23, (float)transformation.M33, (float)transformation.OffsetZ,
        //        (float)transformation.M14, (float)transformation.M24, (float)transformation.M34, (float)transformation.M44
        //    }, new float3(camera));

        //    Rasterize();

        //    VertexShader.Apply(vertexBufferIn, vertexBufferOut, transformation);
        //    PrimitivesAssembler.Assemble(vertexBufferOut, model.Indices, primitivesBuffer, model.Normals);
        //    BackFaceCulling.Cull(primitivesBuffer, camera);

        //    for (int i = 0; i < model.NumberOfTriangles; i ++)
        //    {
        //        var primitive = primitivesBuffer[i];

        //        if (primitive.Visible)
        //        {
        //            rasterizer.DrawTriangle(primitive.v1.Pos, primitive.v2.Pos, primitive.v3.Pos, primitive.v1.Normal, primitive.v2.Normal, primitive.v3.Normal, primitive.v1.Color, primitive.v2.Color, primitive.v3.Color);
        //        }
        //    }

        //    rasterizer.End();
        //}

        public void Redraw()
        {
            if (!isInitialized)
            {
                return;
            }

            bmp.Lock();

            Matrix3D transformation = Matrix3D.Identity;
            transformation.Rotate(new Quaternion(rotation.Axis, -rotation.Angle));
            Point3D camera = transformation.Transform(Camera);
            //rasterizer.light = transformation.Transform(new Point3D(10, 10, -3));

            transformation = Matrix3D.Identity;
            transformation.Rotate(rotation);
            transformation.Translate(-(Vector3D)Camera);
            transformation.Append(projectionMatrix);
            transformation.Translate(new Vector3D(1, 1, 0));
            transformation.Scale(new Vector3D(bmp.Width / 2, bmp.Height / 2, 1));

            SetTransformation(new float[] {
                (float)transformation.M11, (float)transformation.M21, (float)transformation.M31, (float)transformation.OffsetX,
                (float)transformation.M12, (float)transformation.M22, (float)transformation.M32, (float)transformation.OffsetY,
                (float)transformation.M13, (float)transformation.M23, (float)transformation.M33, (float)transformation.OffsetZ,
                (float)transformation.M14, (float)transformation.M24, (float)transformation.M34, (float)transformation.M44
            }, new float3(camera));

            System.Diagnostics.Debug.WriteLine(Rasterize());

            bmp.AddDirtyRect(new Int32Rect(0, 0, (int)bmp.Width, (int)bmp.Height));
            bmp.Unlock();
        }
    }
}

