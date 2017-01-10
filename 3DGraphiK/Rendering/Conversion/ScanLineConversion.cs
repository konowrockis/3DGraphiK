using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media.Media3D;
using GraphiK3D.Rendering.Primitives;

namespace GraphiK3D.Rendering.Conversion
{
    class ScanLineConversion
    {
        private Triangle[] primitivesBuffer;
        
        public static void Convert()
        {

        }

        private static void DrawHalfTriangle(int startY, int endY, ref double startX, ref double endX, double m1, double m2, Triangle primitive)
        {
            for (int y = startY; y < endY; y++)
            {
                if (y < 0)
                {
                    continue;
                }
                if (y >= height)
                {
                    break;
                }

                for (int x = (int)startX; x <= (int)endX; x++)
                {
                    if (x < 0)
                    {
                        continue;
                    }
                    if (x >= width)
                    {
                        break;
                    }

                    var pos = calculateBarycentricCoordinate(new Point3D(x, y, 0), primitive.v1.Pos, primitive.v2.Pos, primitive.v3.Pos);
                    var z = getZAtCoordinate(pos, primitive.v1.Pos, primitive.v2.Pos, primitive.v3.Pos) * 1000;

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

        private static void DrawTriangle(Triangle primitive)
        {
            if (IsPointOutside(primitive.v1.Pos) && IsPointOutside(primitive.v2.Pos) && IsPointOutside(primitive.v3.Pos))
            {
                return;
            }

            //DrawLine(NormalizePos(p1.X, width), NormalizePos(p1.Y, height), NormalizePos(p2.X, width), NormalizePos(p2.Y, height), color);
            //DrawLine(NormalizePos(p2.X, width), NormalizePos(p2.Y, height), NormalizePos(p3.X, width), NormalizePos(p3.Y, height), color);
            //DrawLine(NormalizePos(p3.X, width), NormalizePos(p3.Y, height), NormalizePos(p1.X, width), NormalizePos(p1.Y, height), color);

            var p1 = new Point3D(NormalizePos(primitive.v1.Pos.X, width), NormalizePos(primitive.v1.Pos.Y, height), primitive.v1.Pos.Z);
            var p2 = new Point3D(NormalizePos(primitive.v2.Pos.X, width), NormalizePos(primitive.v2.Pos.Y, height), primitive.v2.Pos.Z);
            var p3 = new Point3D(NormalizePos(primitive.v3.Pos.X, width), NormalizePos(primitive.v3.Pos.Y, height), primitive.v3.Pos.Z);

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
                DrawHalfTriangle((int)top.Y, (int)middle.Y, ref x1, ref x2, m1, m2, primitive);
            }
            else
            {
                x1 = Math.Min(middle.X, top.X);
                x2 = Math.Max(middle.X, top.X);
            }

            m4 = bottom.Y == middle.Y ? 0 : (middle.X - bottom.X) / (middle.Y - bottom.Y);

            m1 = Math.Max(m3, m4);
            m2 = Math.Min(m3, m4);

            DrawHalfTriangle((int)middle.Y, (int)bottom.Y, ref x1, ref x2, m1, m2, primitive);
        }

        private static bool IsPointOutside(Point3D p)
        {
            return p.X < -1 || p.X > 1 || p.Y < -1 || p.Y > 1;
        }

        private static int NormalizePos(double pos, int size)
        {
            return (int)((pos + 1) / 2 * size);
        }

        private static double calculateSignedArea(Point3D p1, Point3D p2, Point3D p3)
        {
            return 0.5 * ((p3.X - p1.X) * (p2.Y - p1.Y) - (p2.X - p1.X) * (p3.Y - p1.Y));
        }

        private static double calculateBarycentricCoordinateValue(Point3D a, Point3D b, Point3D c, double area)
        {
            return calculateSignedArea(a, b, c) / area;
        }

        private static Vector3D calculateBarycentricCoordinate(Point3D point, Point3D p1, Point3D p2, Point3D p3)
        {
            double area = calculateSignedArea(p1, p2, p3);

            double beta = calculateBarycentricCoordinateValue(p1, point, p3, area);
            double gamma = calculateBarycentricCoordinateValue(p1, p2, point, area);
            double alpha = 1.0 - beta - gamma;

            return new Vector3D(alpha, beta, gamma);
        }

        private static double getZAtCoordinate(Vector3D barycentricCoord, Point3D p1, Point3D p2, Point3D p3)
        {
            return -(barycentricCoord.X * p1.Z
                + barycentricCoord.Y * p2.Z
                + barycentricCoord.Z * p3.Z);
        }
    }
}
