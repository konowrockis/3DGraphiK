using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SharpDX;
using SharpDX.Windows;
using D3D9Device = SharpDX.Direct3D9.Device;
using SharpDX.DXGI;
using SharpDX.Direct3D9;
using SharpDX.Mathematics.Interop;
using System.Windows.Media.Media3D;
using System.Runtime.InteropServices;
using static GraphiK3D.Rendering.Renderer;

namespace GraphiK3D
{
    class SimpleD3D9
    {
        public const int Width = 1024;
        public const int Height = 1024;
        public const int Total = Width * Height;

        [DllImport("3DGraphiK.CudaRasterizer.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void Init(IntPtr device);

        [DllImport("3DGraphiK.CudaRasterizer.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr CreateModel(float3[] vertices, float3[] normals, float3[] colors, int[] indices, int numOfVertices, int numOfFaces);

        [DllImport("3DGraphiK.CudaRasterizer.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void SetTransformation(float[] transformation, float3 camera);

        [DllImport("3DGraphiK.CudaRasterizer.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void FreeRasterizer();

        [DllImport("3DGraphiK.CudaRasterizer.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void FreeModel(IntPtr model);

        [DllImport("3DGraphiK.CudaRasterizer.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void Resize(int w, int h, IntPtr backBufSurface);

        [DllImport("3DGraphiK.CudaRasterizer.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void Begin();

        [DllImport("3DGraphiK.CudaRasterizer.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void End();

        [DllImport("3DGraphiK.CudaRasterizer.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void DrawModel(IntPtr model);

        public SimpleD3D9()
        {
        }
        
        public static unsafe void Run()
        {
            var form = new RenderForm("SimpleD3D9 by C#") { ClientSize = new Size(1024, 768) };

            var device = new D3D9Device(
                new Direct3D(),
                0,
                DeviceType.Hardware,
                form.Handle,
                CreateFlags.HardwareVertexProcessing,
                new SharpDX.Direct3D9.PresentParameters(form.ClientSize.Width, form.ClientSize.Height, 
                    SharpDX.Direct3D9.Format.X8R8G8B8, 1, MultisampleType.None, 0, SharpDX.Direct3D9.SwapEffect.Copy, 
                    form.Handle, true, false, SharpDX.Direct3D9.Format.D24X8, 
                    SharpDX.Direct3D9.PresentFlags.LockableBackBuffer, 0, PresentInterval.Immediate));

            Init(device.NativePointer);

            //var view = Matrix.LookAtLH(
            //    new Vector3(0.0f, 3.0f, -2.0f), // the camera position
            //    new Vector3(0.0f, 0.0f, 0.0f),  // the look-at position
            //    new Vector3(0.0f, 1.0f, 0.0f)); // the up direction

            //var proj = Matrix.PerspectiveFovLH(
            //    (float)(Math.PI / 4.0), // the horizontal field of view
            //    1.0f,
            //    1.0f,
            //    100.0f);

            //device.SetTransform(TransformState.View, view);
            //device.SetTransform(TransformState.Projection, proj);
            //device.SetRenderState(RenderState.Lighting, false);

            //var clock = System.Diagnostics.Stopwatch.StartNew();

            var backBuffer = device.GetBackBuffer(0, 0);

            
            Resize(form.Width, form.Height, backBuffer.NativePointer);

            return;

            RenderLoop.Run(form, () =>
            {
                //var b = backBuffer.LockRectangle(LockFlags.None);


                //z.LockRectangle(LockFlags.None);

                //device.UpdateSurface(x, z);

                //z.UnlockRectangle();
                //backBuffer.UnlockRectangle();

                //var time = (float)(clock.Elapsed.TotalMilliseconds) / 300.0f;
                //updater.Update(vbres, time);

                // Now normal D3D9 rendering procedure.
                /*device.Clear(ClearFlags.Target | ClearFlags.ZBuffer, new RawColorBGRA(0, 40, 100, 0), 1.0f, 0);
                device.BeginScene();

                device.EndScene();*/

                //var b = backBuffer.LockRectangle(LockFlags.None);

                

                Begin();
                End();

                //backBuffer.UnlockRectangle();

                device.Present();
            });
            
            device.Dispose();
            form.Dispose();
        }
    }
}
