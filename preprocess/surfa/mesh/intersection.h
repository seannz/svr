#define USE_EPSILON_TEST TRUE
#define EPSILON 0.000001


#define FABS(x) ((float) fabs(x))

#define CROSS(dest, v1, v2)                  \
    dest[0] = v1[1] * v2[2] - v1[2] * v2[1]; \
    dest[1] = v1[2] * v2[0] - v1[0] * v2[2]; \
    dest[2] = v1[0] * v2[1] - v1[1] * v2[0];

#define DOT(v1, v2) (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2])

#define SUB(dest, v1, v2) dest[0] = v1[0] - v2[0]; dest[1] = v1[1] - v2[1]; dest[2] = v1[2] - v2[2];

#define ADD(dest, v1, v2) dest[0] = v1[0] + v2[0]; dest[1] = v1[1] + v2[1]; dest[2] = v1[2] + v2[2];

#define MULT(dest, v, factor) dest[0] = factor * v[0]; dest[1] = factor * v[1]; dest[2] = factor * v[2];

#define SET(dest, src) dest[0] = src[0]; dest[1] = src[1]; dest[2] = src[2];

#define CONTAINS(v, item) ((item == v[0]) || (item == v[1]) || (item == v[2]))

#define SORT(a, b) \
    if (a > b)     \
    {              \
        float c;   \
        c = a;     \
        a = b;     \
        b = c;     \
    }

#define ISECT(VV0, VV1, VV2, D0, D1, D2, isect0, isect1) \
    isect0 = VV0 + (VV1 - VV0) * D0 / (D0 - D1);         \
    isect1 = VV0 + (VV2 - VV0) * D0 / (D0 - D2);

#define COMPUTE_INTERVALS(VV0, VV1, VV2, D0, D1, D2, D0D1, D0D2, isect0, isect1) \
    if (D0D1 > 0.0f)                                         \
    {                                                        \
        ISECT(VV2, VV0, VV1, D2, D0, D1, isect0, isect1);    \
    }                                                        \
    else if (D0D2 > 0.0f)                                    \
    {                                                        \
        ISECT(VV1, VV0, VV2, D1, D0, D2, isect0, isect1);    \
    }                                                        \
    else if (D1 * D2 > 0.0f || D0 != 0.0f)                   \
    {                                                        \
        ISECT(VV0, VV1, VV2, D0, D1, D2, isect0, isect1);    \
    }                                                        \
    else if (D1 != 0.0f)                                     \
    {                                                        \
        ISECT(VV1, VV0, VV2, D1, D0, D2, isect0, isect1);    \
    }                                                        \
    else if (D2 != 0.0f )                                    \
    {                                                        \
        ISECT(VV2, VV0, VV1, D2, D0, D1, isect0, isect1);    \
    }                                                        \
    else                                                     \
    {                                                        \
        return coplanar_tri_tri(N1, V0, V1, V2, U0, U1, U2); \
    }

#define EDGE_EDGE_TEST(V0, U0, U1)                                   \
    Bx = U0[i0] - U1[i0];                                            \
    By = U0[i1] - U1[i1];                                            \
    Cx = V0[i0] - U0[i0];                                            \
    Cy = V0[i1] - U0[i1];                                            \
    f = Ay * Bx - Ax * By;                                           \
    d = By * Cx - Bx * Cy;                                           \
    if ((f > 0 && d >= 0 && d <= f) || (f < 0 && d <= 0 && d >= f))  \
    {                                                                \
        e = Ax * Cy - Ay * Cx;                                       \
        if (f > 0)                                                   \
        {                                                            \
            if (e >= 0 && e <= f) return 1;                          \
        }                                                            \
        else                                                         \
        {                                                            \
            if (e <= 0 && e >= f) return 1;                          \
        }                                                            \
    }

#define EDGE_AGAINST_TRI_EDGES(V0,V1,U0,U1,U2) \
{                                              \
    float Ax, Ay, Bx, By, Cx, Cy, e, d, f;     \
    Ax = V1[i0] - V0[i0];                      \
    Ay = V1[i1] - V0[i1];                      \
    EDGE_EDGE_TEST(V0, U0, U1);                \
    EDGE_EDGE_TEST(V0, U1, U2);                \
    EDGE_EDGE_TEST(V0, U2, U0);                \
}

#define POINT_IN_TRI(V0, U0, U1, U2)   \
{                                      \
    float a, b, c, d0, d1, d2;         \
    a = U1[i1] - U0[i1];               \
    b = -(U1[i0] - U0[i0]);            \
    c = -a * U0[i0] - b * U0[i1];      \
    d0 = a * V0[i0] + b * V0[i1] + c;  \
    a = U2[i1] - U1[i1];               \
    b = -(U2[i0] - U1[i0]);            \
    c = -a * U1[i0] - b * U1[i1];      \
    d1 = a * V0[i0] + b * V0[i1] + c;  \
    a = U0[i1] - U2[i1];               \
    b = -(U0[i0] - U2[i0]);            \
    c = -a * U2[i0] - b * U2[i1];      \
    d2 = a * V0[i0] + b * V0[i1] + c;  \
    if (d0 * d1 > 0.0)                 \
    {                                  \
        if (d0 * d2 > 0.0) return 1;   \
    }                                  \
}

int coplanar_tri_tri(const float N[3],  const float V0[3], const float V1[3], const float V2[3],
                     const float U0[3], const float U1[3], const float U2[3])
{
    float A[3];
    short i0, i1;
    A[0] = FABS(N[0]);
    A[1] = FABS(N[1]);
    A[2] = FABS(N[2]);
    if (A[0] > A[1]) {
        if (A[0] > A[2]) {
            i0 = 1;
            i1 = 2;
        } else {
            i0 = 0;
            i1 = 1;
        }
    } else {
        if (A[2] > A[1]) {
            i0 = 0;
            i1 = 1;
        } else {
            i0 = 0;
            i1 = 2;
        }
    }

    EDGE_AGAINST_TRI_EDGES(V0, V1, U0, U1, U2);
    EDGE_AGAINST_TRI_EDGES(V1, V2, U0, U1, U2);
    EDGE_AGAINST_TRI_EDGES(V2, V0, U0, U1, U2);

    POINT_IN_TRI(V0, U0, U1, U2);
    POINT_IN_TRI(U0, V0, V1, V2);

    return 0;
}

int tri_tri_intersect(const float V0[3], const float V1[3], const float V2[3],
                      const float U0[3], const float U1[3], const float U2[3])
{
    float E1[3], E2[3];
    float N1[3], N2[3], d1, d2;
    float du0, du1, du2, dv0, dv1, dv2;
    float D[3];
    float isect1[2], isect2[2];
    float du0du1, du0du2, dv0dv1, dv0dv2;
    short index;
    float vp0, vp1, vp2;
    float up0, up1, up2;
    float b, c, max;

    SUB(E1, V1, V0);
    SUB(E2, V2, V0);
    CROSS(N1, E1, E2);
    d1 = -DOT(N1, V0);

    du0 = DOT(N1, U0) + d1;
    du1 = DOT(N1, U1) + d1;
    du2 = DOT(N1, U2) + d1;

    #if USE_EPSILON_TEST == TRUE
    if (FABS(du0) < EPSILON) du0 = 0.0;
    if (FABS(du1) < EPSILON) du1 = 0.0;
    if (FABS(du2) < EPSILON) du2 = 0.0;
    #endif
    du0du1 = du0 * du1;
    du0du2 = du0 * du2;

    if (du0du1 > 0.0f && du0du2 > 0.0f)
        return 0;

    SUB(E1, U1, U0);
    SUB(E2, U2, U0);
    CROSS(N2, E1, E2);
    d2 = -DOT(N2, U0);

    dv0 = DOT(N2, V0) + d2;
    dv1 = DOT(N2, V1) + d2;
    dv2 = DOT(N2, V2) + d2;

    #if USE_EPSILON_TEST == TRUE
    if (FABS(dv0) < EPSILON) dv0 = 0.0;
    if (FABS(dv1) < EPSILON) dv1 = 0.0;
    if (FABS(dv2) < EPSILON) dv2 = 0.0;
    #endif

    dv0dv1 = dv0 * dv1;
    dv0dv2 = dv0 * dv2;

    if (dv0dv1 > 0.0f && dv0dv2 > 0.0f)
        return 0;

    CROSS(D, N1, N2);

    max = FABS(D[0]);
    index = 0;
    b = FABS(D[1]);
    c = FABS(D[2]);
    if (b > max) max = b, index = 1;
    if (c > max) max = c, index = 2;

    vp0 = V0[index];
    vp1 = V1[index];
    vp2 = V2[index];

    up0 = U0[index];
    up1 = U1[index];
    up2 = U2[index];

    COMPUTE_INTERVALS(vp0, vp1, vp2, dv0, dv1, dv2, dv0dv1, dv0dv2, isect1[0], isect1[1]);
    COMPUTE_INTERVALS(up0, up1, up2, du0, du1, du2, du0du1, du0du2, isect2[0], isect2[1]);

    SORT(isect1[0], isect1[1]);
    SORT(isect2[0], isect2[1]);

    if (isect1[1] < isect2[0] || isect2[1] < isect1[0]) return 0;
    return 1;
}

void self_intersection_test(const float * vertices, int nverts,
                            const int * faces, int nfaces,
                            const int * selected_faces, int nselected,
                            const int * selected_neighbors, int nneighbors,
                            int * intersecting)
{
    for (int i = 0; i < nselected; i++) {

        int f = selected_faces[i];

        if (intersecting[f] == 1) continue;

        const int * current_face_indices = &faces[f * 3];
        const int * current_neighbors = &selected_neighbors[i * nneighbors];

        for (int n = 0; n < nneighbors; n++) {

            const int neighboring_face = current_neighbors[n];
            if (neighboring_face == f) continue;

            const int * neighboring_face_indices = &faces[neighboring_face * 3];
            if (CONTAINS(neighboring_face_indices, current_face_indices[0]) ||
                CONTAINS(neighboring_face_indices, current_face_indices[1]) ||
                CONTAINS(neighboring_face_indices, current_face_indices[2])) continue;

            int intersect = tri_tri_intersect(
                &vertices[current_face_indices[0] * 3],
                &vertices[current_face_indices[1] * 3],
                &vertices[current_face_indices[2] * 3],
                &vertices[neighboring_face_indices[0] * 3],
                &vertices[neighboring_face_indices[1] * 3],
                &vertices[neighboring_face_indices[2] * 3]);

            if (intersect == 1) {
                intersecting[f] = 1;
                intersecting[neighboring_face] = 1;
                break;
            }
        }
    }
};
