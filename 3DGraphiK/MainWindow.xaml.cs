using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Media.Media3D;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Windows.Threading;
using GraphiK3D.Models;
using GraphiK3D.Rendering;

namespace _3DGraphiK
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private Renderer renderer;

        public MainWindow()
        {
            renderer = new Renderer();

            InitializeComponent();

            Task t = new Task(() =>
             {
                 while (true)
                 {
                     try
                     {
                         Dispatcher.Invoke(() =>
                         {
                             renderer.Redraw();
                             UpdateFps();
                         });
                         Thread.Sleep(5);
                     }
                     catch { break; }
                 }
             });
            t.Start();
        }

        private DateTime lastDate = DateTime.Now;
        private int framesCount = 0;

        private void UpdateFps()
        {
            if ((DateTime.Now - lastDate).TotalMilliseconds > 500)
            {
                Title = (Math.Round(framesCount * 10000 / (DateTime.Now - lastDate).TotalMilliseconds) / 10).ToString();
                lastDate = DateTime.Now;
                framesCount = 1;
            }
            else
            {
                framesCount++;
            }
        }


        protected override void OnRenderSizeChanged(SizeChangedInfo sizeInfo)
        {
            var size = SceneWrapper.RenderSize;
            Scene.Source = renderer.CreateRasterizer((int)size.Width, (int)size.Height);

            renderer.Redraw();
            base.OnRenderSizeChanged(sizeInfo);
        }

        private bool isMouseDown = false;
        private Point lastMouseLocation;

        protected override void OnMouseDown(MouseButtonEventArgs e)
        {
            isMouseDown = true;
            lastMouseLocation = e.GetPosition(this);
        }

        protected override void OnMouseUp(MouseButtonEventArgs e)
        {
            isMouseDown = false;
        }

        protected override void OnMouseMove(MouseEventArgs e)
        {
            if (isMouseDown)
            {
                Point location = e.GetPosition(this);

                var diff = lastMouseLocation - location;

                renderer.rotation = Quaternion.Multiply(new Quaternion(new Vector3D(0, 1, 0), -diff.X), renderer.rotation);
                renderer.rotation = Quaternion.Multiply(new Quaternion(new Vector3D(1, 0, 0), diff.Y), renderer.rotation);

                lastMouseLocation = location;
            }
        }
    }
}
