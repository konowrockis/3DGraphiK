using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Windows.Media.Media3D;

namespace GraphiK3D.Models
{
    static class ModelLoader
    {
        private enum LoaderMode
        {
            Id, Vertices, Indices
        }

        private static CultureInfo Culture;

        static ModelLoader()
        {
            Culture = (CultureInfo)CultureInfo.CurrentCulture.Clone();
            Culture.NumberFormat.CurrencyDecimalSeparator = ".";
        }

        public static Model Load(string fileName)
        {
            StreamReader reader = new StreamReader(fileName);

            LoaderMode mode = LoaderMode.Id;

            int verticesCount = 0, facesCount = 0, edgesCount = 0;
            List<Point3D> vertices = new List<Point3D>();
            List<int> indices = new List<int>();

            while (!reader.EndOfStream)
            {
                string line = reader.ReadLine().ToLower();
                if (line.Trim() == string.Empty || line.StartsWith("#"))
                {
                    continue;
                }

                var vals = line.Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);

                switch (mode)
                {
                    case LoaderMode.Id:
                    if (line == "off")
                    {
                        continue;
                    }
                    else
                    {
                        verticesCount = GetInt(vals[0]);
                        facesCount = GetInt(vals[1]);
                        edgesCount = GetInt(vals[2]);

                        mode = LoaderMode.Vertices;
                    }
                    break;

                    case LoaderMode.Vertices:

                    float x = GetFloat(vals[0]);
                    float y = GetFloat(vals[1]);
                    float z = GetFloat(vals[2]);

                    vertices.Add(new Point3D(x, y, z));

                    if (--verticesCount == 0)
                    {
                        mode = LoaderMode.Indices;
                    }
                    break;

                    case LoaderMode.Indices:
                    indices.AddRange(new int[] { GetInt(vals[1]), GetInt(vals[2]), GetInt(vals[3]) });
                    break;
                }
            }

            return new Model(vertices.ToArray(), indices.ToArray());
        }

        private static float GetFloat(string val)
        {
            return float.Parse(val, NumberStyles.Any, Culture);
        }

        private static int GetInt(string val)
        {
            return int.Parse(val);
        }
    }
}

