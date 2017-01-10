using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media.Media3D;

namespace GraphiK3D.Models
{
    class Model
    {
        public Point3D[] Vertices;
        public int[] Indices;
        public Vector3D[] Colors;

        public Vector3D[] Normals;
        public Vector3D[] NormalVertices;

        public int NumberOfTriangles;

        public Model(Point3D[] vertices, int[] indices)
        {
            Vertices = vertices;
            Indices = indices;
            NumberOfTriangles = indices.Length / 3;

            Normals = ComputeNormals(vertices, indices, NumberOfTriangles);
            NormalVertices = ComputeNormalVertices(Normals, indices, vertices.Length);

            Colors = CreateColors(vertices.Length);
        }

        private static Vector3D[] ComputeNormals(Point3D[] vertices, int[] indices, int numberOfTriangles)
        {
            Vector3D[] normals = new Vector3D[numberOfTriangles];
            
            for (int i = 0; i < numberOfTriangles; i++)
            {
                var v1 = vertices[indices[i * 3 + 1]] - vertices[indices[i * 3]];
                var v2 = vertices[indices[i * 3 + 2]] - vertices[indices[i * 3]];

                normals[i] = Vector3D.CrossProduct(v1, v2);
                normals[i].Normalize();
            }

            return normals;
        }

        private static Vector3D[] CreateColors(int numberOfVertices)
        {
            Vector3D[] colors = new Vector3D[numberOfVertices];
            Random r = new Random();

            for (int i = 0; i < numberOfVertices; i++)
            {
                colors[i] = new Vector3D(r.NextDouble(), r.NextDouble(), r.NextDouble());
                //colors[i] = new Vector3D(0.85, 0.2, 0.2);
            }

            return colors;
        }

        private static Vector3D[] ComputeNormalVertices(Vector3D[] normals, int[] indices, int numberOfVertices)
        {
            Vector3D[] normalVertices = new Vector3D[numberOfVertices];

            for (int i = 0; i < numberOfVertices; i++)
            {
                Vector3D v = new Vector3D(0, 0, 0);
                int count = 0;

                for (int j = 0; j < indices.Length; j++)
                {
                    if (indices[j] == i)
                    {
                        v += normals[j / 3];
                        count++;
                    }
                }

                if (count > 1)
                {
                    normalVertices[i] = v / count;
                    normalVertices[i].Normalize();
                }
            }

            return normalVertices;
        }
    }
}
