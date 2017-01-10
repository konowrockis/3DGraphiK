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

namespace GraphiK3D
{
    class SimpleD3D9
    {
        public const int Width = 1024;
        public const int Height = 1024;
        public const int Total = Width * Height;

        public SimpleD3D9()
        {
        }

        //[Kernel]
        //public void Kernel(deviceptr<float4> pos, float time)
        //{
        //    var x = blockIdx.x * blockDim.x + threadIdx.x;
        //    var y = blockIdx.y * blockDim.y + threadIdx.y;

        //    var u = ((float)x) / ((float)Width);
        //    var v = ((float)y) / ((float)Height);
        //    u = u * 2.0f - 1.0f;
        //    v = v * 2.0f - 1.0f;

        //    const float freq = 4.0f;
        //    var w = LibDevice.__nv_sinf(u * freq + time) * LibDevice.__nv_cosf(v * freq + time) * 0.5f;

        //    pos[y * Width + x] = new float4(u, w, v, LibDevice.__nv_uint_as_float(0xff00ff00));
        //}

        //unsafe public void Update(IntPtr vbRes, float time)
        //{
        //    // 1. map resource to cuda space, means lock to cuda space
        //    var vbRes1 = vbRes;
        //    CUDAInterop.cuSafeCall(CUDAInterop.cuGraphicsMapResources(1, &vbRes1, IntPtr.Zero));

        //    // 2. get memory pointer from mapped resource
        //    var vbPtr = IntPtr.Zero;
        //    var vbSize = IntPtr.Zero;
        //    CUDAInterop.cuSafeCall(CUDAInterop.cuGraphicsResourceGetMappedPointer(&vbPtr, &vbSize, vbRes1));

        //    // 3. create device pointer, and run the kernel
        //    var pos = new deviceptr<float4>(vbPtr);
        //    GPULaunch(Kernel, LaunchParam, pos, time);

        //    // 4. unmap resource, means unlock, so that DirectX can then use it again
        //    CUDAInterop.cuSafeCall(CUDAInterop.cuGraphicsUnmapResources(1u, &vbRes1, IntPtr.Zero));
        //}

        //static void UnregisterVerticesResource(IntPtr res)
        //{
        //    CUDAInterop.cuSafeCall(CUDAInterop.cuGraphicsUnregisterResource(res));
        //}

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
                    form.Handle, true, true, SharpDX.Direct3D9.Format.D24X8, 
                    SharpDX.Direct3D9.PresentFlags.LockableBackBuffer, 0, PresentInterval.Immediate));
            
            var vertices = new VertexBuffer(device, Utilities.SizeOf<Vector3D>() * Total, SharpDX.Direct3D9.Usage.WriteOnly,
                VertexFormat.None, Pool.Default);

            var vertexElems = new[]
            {
                new VertexElement(0, 0, DeclarationType.Float3, DeclarationMethod.Default, DeclarationUsage.Position, 0),
                new VertexElement(0, 12, DeclarationType.Ubyte4, DeclarationMethod.Default, DeclarationUsage.Color, 0),
                VertexElement.VertexDeclarationEnd
            };

            var vertexDecl = new VertexDeclaration(device, vertexElems);

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
            Texture t = new Texture(device, form.Width, form.Height, 1, SharpDX.Direct3D9.Usage.None, SharpDX.Direct3D9.Format.A8R8G8B8, Pool.Default);
            t = Texture.FromFile(device, "audyty-i-optymalizacja-kosztowa.jpg");

            Sprite sprite = null;
            sprite = new SharpDX.Direct3D9.Sprite(device);

            var x = device.GetBackBuffer(0, 0);
            var z = SharpDX.Direct3D9.Surface.CreateOffscreenPlain(device, form.Width, form.Height, x.Description.Format, Pool.SystemMemory);

            //z.NativePointer
            //device.GetRenderTargetData(x, z);

            RenderLoop.Run(form, () =>
            {
                var b = x.LockRectangle(LockFlags.None);
                //z.LockRectangle(LockFlags.None);

                //device.UpdateSurface(x, z);

                //z.UnlockRectangle();
                x.UnlockRectangle();

                //var time = (float)(clock.Elapsed.TotalMilliseconds) / 300.0f;
                //updater.Update(vbres, time);

                // Now normal D3D9 rendering procedure.
                /*device.Clear(ClearFlags.Target | ClearFlags.ZBuffer, new RawColorBGRA(0, 40, 100, 0), 1.0f, 0);
                device.BeginScene();

                //device.VertexDeclaration = vertexDecl;
                //device.SetStreamSource(0, vertices, 0, Utilities.SizeOf<Vector4>());
                // we use PointList as the graphics primitives
                //device.DrawPrimitives(SharpDX.Direct3D9.PrimitiveType.PointList, 0, Total);

                sprite.Begin(SharpDX.Direct3D9.SpriteFlags.None);
                sprite.Draw(t, new RawColorBGRA(0xFF, 0xFF, 0xFF, 0x00));

                sprite.End();

                device.EndScene();*/

                device.Present();
            });

            //UnregisterVerticesResource(vbres);

            //updater.Dispose();
            //worker.Dispose();
            //vertexDecl.Dispose();
            //vertices.Dispose();
            device.Dispose();
            form.Dispose();
        }
    }
}
