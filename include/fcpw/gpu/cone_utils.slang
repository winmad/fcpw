implementing fcpw;
__include ray;
__include bounding_volumes;

/**
 * Returns a maximal angular similarity to the segment relative to the cone,
 * as well as the parameterization "t" along the segment to the corresponding point.
 * Similarity is expressed as s = dot(d, ∆) |dot(d, ∆)| / |∆|^2
 * @param[in] a  - the apex of the cone
 * @param[in] d  - the direction of the cone, perpendicular to it's base.
 * @param[in] v0 - the starting point of the segment
 * @param[in] v1 - the ending point of the segment
 * @return {similarity, the t to lerp between v0 and v1 to retrieve the tangent point}
 */
float2 minAngleToSegment(float3 a, float3 d, float3 v0, float3 v1)
{
    //  ∆0 and ∆1 are vectors from segment to the apex
    float3 d0 = v0 - a;
    float3 d1 = v1 - a;

    // α - angle between d and ∆0
    // β - angle between d and ∆1
    // γ - angle between ∆0 and ∆1
    // θ - minimizing angle between D and the segment

    // Values are:

    // |∆0|^2
    float dd00 = dot(d0, d0);

    // |∆1|^2
    float dd11 = dot(d1, d1);

    // |∆0||D|cos(α)
    float dd0d = dot(d0, d);

    // |∆1||D|cos(β)
    float dd1d = dot(d1, d);

    // |∆0||∆1|cos(γ)
    float dd01 = dot(d0, d1);

    // Vector perpendicular to triangle from edge to apex, scaled by parallelogram.
    // Equivalent to great circle normal connecting ∆0 and ∆1, but scaled.
    // |∆0x∆1| = |∆0||∆1|sin(γ)
    float3 n = cross(d0, d1);

    // |N|^2
    float dnn = dot(n, n);

    // |D||N|cos(θ)
    float ddn = dot(d, n);

    // Segment is degenerate, so the great circle normal becomes numerically unstable.
    if (abs(ddn) < 1e-30f) {
        ddn = sign(ddn) * 1e-30f;
    }

    // Project d onto plane containing the triangle.
    // (aka, project point onto great circle containing arc from v0 to v1)
    // p = D - (D·N)/(N·N) * N
    float3 d2 = d - (ddn / dnn) * n;

    // Find planes perpendicular to great circle containing the arc
    // (using triple vector product rule to avoid cross products)
    //   A × (B × C) = B (A · C) − C (A · B),
    //   (A × B) × C = C (A · B) - B (A · C),
    //   N0 = Nx∆0 = (∆0x∆1)x∆0 = ∆1(∆0·∆0) - ∆0(∆0·∆1)
    //   N1 = Nx∆1 = (∆0x∆1)x∆1 = ∆0(∆1·∆1) - ∆1(∆0·∆1)
    float3 n0 = d1 * dd00 - d0 * dd01;
    float3 n1 = d0 * dd11 - d1 * dd01;

    // Test if projection falls outside the arc
    if (dot(d2, n0) < 0.f || dot(d2, n1) < 0.f) {
        // Compute angular similarity measure to the end points
        // similarity = dot(d, ∆) |dot(d, ∆)| / |∆|^2
        float sa = dd0d * abs(dd0d) / dd00;
        float sb = dd1d * abs(dd1d) / dd11;

        // Pick the closer of the two ends
        if (sa > sb) return float2(sa, 0.);
        else return float2(sb, 1.);
    }

    // Projection is on the triangle / within the arc.
    // Compute the barycentrics of the projected point within
    // that triangle to recover the "t" parameter interpolating the edge.
    float d20 = dot(d2, d0);
    float d21 = dot(d2, d1);

    float d22 = dot(d2, d2);
    float area = (dd00 * dd11 - dd01 * dd01);
    float v = ((dd11 * d20) - (dd01 * d21)) / area;
    float w = ((dd00 * d21) - (dd01 * d20)) / area;
    float t = w / (w + v);

    // similarity = dot(d, ∆) |dot(d, ∆)| / |∆|^2
    float dd2 = dot(d, d2);
    float s = dd2 * abs(dd2) / d22;
    return float2(s, t);
}

/**
 * Given an angular similarity, returns the corresponding angle in radians.
 *   s - the similarity measure, s = (dot(d, ∆) |dot(d, ∆)| / |∆|^2)
 */
float angularSimilarityToAngle(float s) {
    s = clamp(s, -1.f, 1.f);
    if (0.f <= s && s <= 1.f)
        return acos(clamp(2.f * s - 1.0, -1.0, 1.0)) / 2.0;
    else // if (-1.f <= s)
        return (acos(clamp(2.f * s + 1.0, -1.0, 1.0)) / 2.0) + radians(90.);
}

void get_closest_face(float3 o, float3 d, float3 d_inv, float3 aabb_lo, float3 aabb_hi, out float3[4] p)
{
    float3 lo = (aabb_lo - o) * d_inv;
    float3 hi = (aabb_hi - o) * d_inv;
    float3 nr = min(lo, hi);
    float3 fr = max(lo, hi);

    int axis = 0;
    if (fr.x <= fr.y)
        axis = (fr.x <= fr.z) ? 0 : 2;
    else
        axis = (fr.y <= fr.z) ? 1 : 2;

    if (axis == 0)
    {
        p[0] = float3(fr.x, nr.y, nr.z);
        p[1] = float3(fr.x, fr.y, nr.z);
        p[2] = float3(fr.x, fr.y, fr.z);
        p[3] = float3(fr.x, nr.y, fr.z);
    }
    else if (axis == 1)
    {
        p[0] = float3(nr.x, fr.y, nr.z);
        p[1] = float3(fr.x, fr.y, nr.z);
        p[2] = float3(fr.x, fr.y, fr.z);
        p[3] = float3(nr.x, fr.y, fr.z);
    }
    else
    {
        p[0] = float3(nr.x, nr.y, fr.z);
        p[1] = float3(fr.x, nr.y, fr.z);
        p[2] = float3(fr.x, fr.y, fr.z);
        p[3] = float3(nr.x, fr.y, fr.z);
    }

    for (int i = 0; i < 4; i++)
        p[i] = o + p[i] * d;
}

int get_closest_edge(float3 o, float3 d, float3[4] p)
{
    // Fix orientation.
    float3[4] q;
    for (int i = 0; i < 4; i++)
        q[i] = p[i] - o;

    bool change_polygon_orientation = false;
    float3 orient = cross(q[1] - q[0], q[2] - q[0]);
    if (dot(q[0], orient) > 0.f)
    {
        change_polygon_orientation = true;
        float3 tmp = q[1];
        q[1] = q[3];
        q[3] = tmp;
    }

    // Find the closest edge.
    int edge_index = -1;
    float max_dot = -1.f;
    for (int i = 0; i < 4; i++)
    {
        float3 p0 = q[i];
        float3 p1 = q[(i + 1) % 4];

        float3 n = cross(p0, p1);
        float length_n = length(n);
        float dot_val = dot(n, d) / length_n;
        if (dot_val > max_dot)
        {
            max_dot = dot_val;
            edge_index = (change_polygon_orientation ? 3 - i : i);
        }
    }
    return edge_index;
}

float minAngleToAABB(float3 o, float3 d, BoundingBox bbox)
{
    Ray ray = Ray(o, d, FLT_MAX);
    float tMin, tMax;
    if (bbox.intersect(ray, tMin, tMax))
        return 1.f;

    float3 p[4];
    get_closest_face(o, d, ray.dInv, bbox.pMin, bbox.pMax, p);

    float maxCos = 0.f;
#if 0
    for (uint k = 0; k < 4; k++)
    {
        float3 p0 = p[k];
        float3 p1 = p[(k + 1) % 4];

        float2 res = minAngleToSegment(o, d, p0, p1);
        maxCos = max(maxCos, res.x);
    }
#else
    int edge_index = get_closest_edge(o, d, p);
    if (edge_index != -1)
    {
        float3 p0 = p[edge_index];
        float3 p1 = p[(edge_index + 1) % 4];

        float2 res = minAngleToSegment(o, d, p0, p1);
        maxCos = max(maxCos, res.x);
    }
#endif

    return maxCos;
}
