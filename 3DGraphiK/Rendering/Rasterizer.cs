using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Media.Media3D;

namespace GraphiK3D.Rendering
{
    public unsafe class Rasterizer
    {
        private WriteableBitmap canvas;
        private int* pBackBuffer;
        private int height;
        private int width;
        private double[,] zBuffer;

        public int this[double x, double y]
        {
            get
            {
                if (x % 1 == 0 && y % 1 == 0)
                {
                    return this[(int)x, (int)y];
                }
                else if (x % 1 == 0 || y % 1 == 0)
                {
                    int v1, v2;
                    double frac;

                    if (x % 1 == 0)
                    {
                        v1 = this[(int)x, (int)Math.Floor(y)];
                        v2 = this[(int)x, (int)Math.Ceiling(y)];
                        frac = y % 1;
                    }
                    else
                    {
                        v1 = this[(int)Math.Floor(x), (int)y];
                        v2 = this[(int)Math.Ceiling(x), (int)y];
                        frac = x % 1;
                    }

                    double r = (v1 >> 16 & 0xFF) * (1 - frac) + (v2 >> 16 & 0xFF) * frac;
                    double g = (v1 >> 8 & 0xFF) * (1 - frac) + (v2 >> 8 & 0xFF) * frac;
                    double b = (v1 >> 0 & 0xFF) * (1 - frac) + (v2 >> 0 & 0xFF) * frac;

                    return Clamp(r) << 16 | Clamp(g) << 8 | Clamp(b);
                }
                else
                {
                    int v00 = this[(int)Math.Floor(x), (int)Math.Floor(y)];
                    int v01 = this[(int)Math.Floor(x), (int)Math.Ceiling(y)];
                    int v10 = this[(int)Math.Ceiling(x), (int)Math.Floor(y)];
                    int v11 = this[(int)Math.Ceiling(x), (int)Math.Ceiling(y)];

                    double fracx = x % 1;
                    double fracy = y % 1;

                    double r = ((v00 >> 16 & 0xFF) * (1 - fracx) + (v10 >> 16 & 0xFF) * fracx) * (1 - fracy) + ((v01 >> 16 & 0xFF) * (1 - fracx) + (v11 >> 16 & 0xFF) * fracx) * fracy;
                    double g = ((v00 >> 8 & 0xFF) * (1 - fracx) + (v10 >> 8 & 0xFF) * fracx) * (1 - fracy) + ((v01 >> 8 & 0xFF) * (1 - fracx) + (v11 >> 8 & 0xFF) * fracx) * fracy;
                    double b = ((v00 & 0xFF) * (1 - fracx) + (v10 & 0xFF) * fracx) * (1 - fracy) + ((v01 & 0xFF) * (1 - fracx) + (v11 & 0xFF) * fracx) * fracy;

                    return Clamp(r) << 16 | Clamp(g) << 8 | Clamp(b);
                }
            }
        }

        public int this[int x, int y]
        {
            get
            {
                if (x < 0 || x >= width || y < 0 || y >= height)
                {
                    return 0;
                }

                return pBackBuffer[y * width + x];
            }
            set
            {
                if (x < 0 || x >= width || y < 0 || y >= height)
                {
                    return;
                }

                pBackBuffer[y * width + x] = value;
            }
        }

        public Rasterizer(int width, int height)
        {
            SetCanvasSize(width, height);
        }

        public void SetCanvasSize(int width, int height)
        {
            canvas = new WriteableBitmap(width, height, 96, 96, PixelFormats.Bgr32, null);
            zBuffer = new double[width, height];
            pBackBuffer = (int*)canvas.BackBuffer;
            this.height = canvas.PixelHeight;
            this.width = canvas.PixelWidth;
        }

        public void Begin()
        {
            canvas.Lock();
        }

        public void End()
        {
            canvas.AddDirtyRect(new Int32Rect(0, 0, canvas.PixelWidth, canvas.PixelHeight));
            canvas.Unlock();
        }

        public ImageSource GetSource()
        {
            return canvas;
        }

        Random r = new Random();

        private double calculateSignedArea(Point3D p1, Point3D p2, Point3D p3)
        {
            return 0.5 * ((p3.X - p1.X) * (p2.Y - p1.Y) - (p2.X - p1.X) * (p3.Y - p1.Y));
        }

        double calculateBarycentricCoordinateValue(Point3D a, Point3D b, Point3D c, double area)
        {
            return calculateSignedArea(a, b, c) / area;
        }

        private Vector3D calculateBarycentricCoordinate(Point3D point, Point3D p1, Point3D p2, Point3D p3)
        {
            double area = calculateSignedArea(p1, p2, p3);

            double beta = calculateBarycentricCoordinateValue(p1, point, p3, area);
            double gamma = calculateBarycentricCoordinateValue(p1, p2, point, area);
            double alpha = 1.0 - beta - gamma;

            return new Vector3D(alpha, beta, gamma);
        }

        double getZAtCoordinate(Vector3D barycentricCoord, Point3D p1, Point3D p2, Point3D p3)
        {
            return -(barycentricCoord.X * p1.Z
                + barycentricCoord.Y * p2.Z
                + barycentricCoord.Z * p3.Z);
        }

        public Point3D light;

        private void DrawHalfTriangle(int startY, int endY, ref double startX, ref double endX, double m1, double m2,
            Point3D p1, Point3D p2, Point3D p3, Vector3D n1, Vector3D n2, Vector3D n3, Vector3D c1, Vector3D c2, Vector3D c3)
        {
            for (int y = startY; y < endY; y++)
            {
                if (y < 0)
                {
                    continue;
                }
                if (y > height)
                {
                    break;
                }

                for (int x = (int)startX; x <= (int)endX; x++)
                {
                    if (x < 0)
                    {
                        continue;
                    }
                    if (x > width)
                    {
                        break;
                    }

                    var pos = calculateBarycentricCoordinate(new Point3D(x, y, 0), p1, p2, p3);
                    var z = getZAtCoordinate(pos, p1, p2, p3) * 1000;

                    if (zBuffer[x, y] > z)
                    {
                        zBuffer[x, y] = z;

                        var color = pos.X * c1 + pos.Y * c2 + pos.Z * c3;
                        var normal = pos.X * n1 + pos.Y * n2 + pos.Z * n3;

                        /*normal = (normal + new Vector3D(1, 1, 1)) / 2;

                        double ks = 0.3, kd = 0.7, ka = 0;

                        Vector3D l = (Vector3D)(light - v);
                        l.Normalize();

                        Vector3D li = color;
                        Vector3D la = new Vector3D(1, 1, 1);

                        Vector3D r = 2 * Vector3D.DotProduct(l, normal) * normal - l;

                        r.Normalize();
                        normal.Normalize();

                        color = ka * la + li * (kd * Vector3D.DotProduct(normal, l) + ks * Math.Pow(Vector3D.DotProduct(r, v), 128));*/

                        DrawDot(x, y, color);
                    }
                }

                startX += m1;
                endX += m2;
            }
        }

        public void DrawTriangle(Point3D p1, Point3D p2, Point3D p3, Vector3D n1, Vector3D n2, Vector3D n3, Vector3D c1, Vector3D c2, Vector3D c3)
        {
            if (IsPointOutside(p1) && IsPointOutside(p2) && IsPointOutside(p3))
            {
                return;
            }

            DrawLine(NormalizePos(p1.X, width), NormalizePos(p1.Y, height), NormalizePos(p2.X, width), NormalizePos(p2.Y, height), 0xFFFFFF);
            DrawLine(NormalizePos(p2.X, width), NormalizePos(p2.Y, height), NormalizePos(p3.X, width), NormalizePos(p3.Y, height), 0xFFFFFF);
            DrawLine(NormalizePos(p3.X, width), NormalizePos(p3.Y, height), NormalizePos(p1.X, width), NormalizePos(p1.Y, height), 0xFFFFFF);

            return;

            p1 = new Point3D(NormalizePos(p1.X, width), NormalizePos(p1.Y, height), p1.Z);
            p2 = new Point3D(NormalizePos(p2.X, width), NormalizePos(p2.Y, height), p2.Z);
            p3 = new Point3D(NormalizePos(p3.X, width), NormalizePos(p3.Y, height), p3.Z);

            var pts = new List<Point3D> { p1, p2, p3 }.OrderBy(p => p.Y);
            Point3D top = pts.First(), bottom = pts.Last(), middle = pts.Skip(1).First();

            double m3 = top.Y == bottom.Y ? 0 : (top.X - bottom.X) / (top.Y - bottom.Y);
            double m4 = top.Y == middle.Y ? 0 : (top.X - middle.X) / (top.Y - middle.Y);

            double m1 = Math.Min(m3, m4);
            double m2 = Math.Max(m3, m4);

            double x1 = top.X;
            double x2 = top.X;

            if (top.Y != middle.Y)
            {
                DrawHalfTriangle((int)top.Y, (int)middle.Y, ref x1, ref x2, m1, m2, p1, p2, p3, n1, n2, n3, c1, c2, c3);
            }
            else
            {
                x1 = Math.Min(middle.X, top.X);
                x2 = Math.Max(middle.X, top.X);
            }

            m4 = bottom.Y == middle.Y ? 0 : (middle.X - bottom.X) / (middle.Y - bottom.Y);

            m1 = Math.Max(m3, m4);
            m2 = Math.Min(m3, m4);

            DrawHalfTriangle((int)middle.Y, (int)bottom.Y, ref x1, ref x2, m1, m2, p1, p2, p3, n1, n2, n3, c1, c2, c3);
        }

        private int NormalizePos(double pos, int size)
        {
            return (int)((pos + 1) / 2 * size);
        }

        private bool IsPointOutside(Point3D p)
        {
            return p.X < -1 || p.X > 1 || p.Y < -1 || p.Y > 1;
        }

        private void DrawDot(int x, int y, int color)
        {
            this[x, y] = color;
        }

        private void DrawDot(int x, int y, Vector3D color)
        {
            this[x, y] = 
                Clamp(color.X * 255) << 16 | 
                Clamp(color.Y * 255) << 8 | 
                Clamp(color.Z * 255);
        }

        private void DrawLine(int x1, int y1, int x2, int y2, int color)
        {
            int dx = Math.Abs(x2 - x1);
            int dy = Math.Abs(y2 - y1);
            int sx = (x1 < x2) ? 1 : -1;
            int sy = (y1 < y2) ? 1 : -1;
            int err = dx - dy;

            DrawDot(x1, y1, color);

            while (!((x1 == x2) && (y1 == y2)))
            {
                int e2 = err << 1;
                if (e2 > -dy)
                {
                    err -= dy;
                    x1 += sx;
                }
                if (e2 < dx)
                {
                    err += dx;
                    y1 += sy;
                }

                DrawDot(x1, y1, color);
            }
        }

        public void Clear(int color)
        {
            unsafe
            {
                for (int i = 0; i < height * width; i++)
                {
                    pBackBuffer[i] = color;
                }

                for (int x = 0; x < width; x++)
                {
                    for (int y = 0; y < height; y++)
                    {
                        zBuffer[x, y] = double.PositiveInfinity;
                    }
                }
            }
        }

        private int CombineColor(int r, int g, int b)
        {
            return Clamp(r) << 16 | Clamp(g) << 8 | Clamp(b);
        }

        private void ParseColor(int color, out int r, out int g, out int b)
        {
            r = (color & 0xFF0000) >> 16;
            g = (color & 0x00FF00) >> 8;
            b = (color & 0x0000FF);
        }

        private int CombineColor(double r, double g, double b)
        {
            return Clamp(r) << 16 | Clamp(g) << 8 | Clamp(b);
        }

        private byte Clamp(double c, double min = 0, double max = 255)
        {
            return (byte)Math.Min(max, Math.Max(min, c));
        }

        private void CopyToRaster(int startX, int startY, int[,] buffer)
        {
            for (int x = 0; x < buffer.GetLength(0); x++)
            {
                for (int y = 0; y < buffer.GetLength(1); y++)
                {
                    this[startX + x, startY + y] = buffer[x, y];
                }
            }
        }
    }
}