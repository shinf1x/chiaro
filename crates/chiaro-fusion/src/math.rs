//! Small fixed-size linear algebra used by the camera model. Kept explicit and
//! dependency-free so the geometry reads like the equations in the docs.

pub type Vec2 = [f64; 2];
pub type Vec3 = [f64; 3];
/// Row-major 3x3 matrix: `m[row][column]`.
pub type Mat3 = [[f64; 3]; 3];

pub const IDENTITY: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

pub fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn scale(a: Vec3, s: f64) -> Vec3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

pub fn norm(a: Vec3) -> f64 {
    dot(a, a).sqrt()
}

pub fn normalize(a: Vec3) -> Vec3 {
    let n = norm(a);
    if n == 0.0 { a } else { scale(a, 1.0 / n) }
}

pub fn mul_vec(m: &Mat3, v: Vec3) -> Vec3 {
    [dot(m[0], v), dot(m[1], v), dot(m[2], v)]
}

pub fn mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

pub fn transpose(m: &Mat3) -> Mat3 {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

pub fn determinant(m: &Mat3) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

pub fn inverse(m: &Mat3) -> Option<Mat3> {
    let det = determinant(m);
    if det.abs() < 1e-300 {
        return None;
    }
    let inv_det = 1.0 / det;
    let c =
        |r0: usize, c0: usize, r1: usize, c1: usize| m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0];
    Some([
        [
            c(1, 1, 2, 2) * inv_det,
            -c(0, 1, 2, 2) * inv_det,
            c(0, 1, 1, 2) * inv_det,
        ],
        [
            -c(1, 0, 2, 2) * inv_det,
            c(0, 0, 2, 2) * inv_det,
            -c(0, 0, 1, 2) * inv_det,
        ],
        [
            c(1, 0, 2, 1) * inv_det,
            -c(0, 0, 2, 1) * inv_det,
            c(0, 0, 1, 1) * inv_det,
        ],
    ])
}

/// Rotation of `angle_radians` about `axis` (Rodrigues' formula).
pub fn rotation_about_axis(axis: Vec3, angle_radians: f64) -> Mat3 {
    let [x, y, z] = normalize(axis);
    let (s, c) = angle_radians.sin_cos();
    let t = 1.0 - c;
    [
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ]
}

/// Rotation from an axis-angle vector whose length is the angle in radians.
pub fn rotation_from_axis_angle(vector: Vec3) -> Mat3 {
    let angle = norm(vector);
    if angle < 1e-18 {
        return IDENTITY;
    }
    rotation_about_axis(vector, angle)
}

/// Householder reflection across the plane with unit normal `normal`.
pub fn reflection(normal: Vec3) -> Mat3 {
    let [x, y, z] = normalize(normal);
    [
        [1.0 - 2.0 * x * x, -2.0 * x * y, -2.0 * x * z],
        [-2.0 * x * y, 1.0 - 2.0 * y * y, -2.0 * y * z],
        [-2.0 * x * z, -2.0 * y * z, 1.0 - 2.0 * z * z],
    ]
}

/// Apply a 3x3 homography to a 2-D point. Returns `None` at infinity.
pub fn apply_homography(h: &Mat3, p: Vec2) -> Option<Vec2> {
    let v = mul_vec(h, [p[0], p[1], 1.0]);
    if v[2].abs() < 1e-12 {
        None
    } else {
        Some([v[0] / v[2], v[1] / v[2]])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_and_rotation_are_consistent() {
        let r = rotation_about_axis([0.3, -0.2, 0.9], 0.7);
        let inv = inverse(&r).unwrap();
        let product = mul(&r, &inv);
        for i in 0..3 {
            for j in 0..3 {
                assert!((product[i][j] - IDENTITY[i][j]).abs() < 1e-12);
                assert!((inv[i][j] - transpose(&r)[i][j]).abs() < 1e-12);
            }
        }
        assert!((determinant(&r) - 1.0).abs() < 1e-12);
        let m = reflection([0.0, 0.0, 1.0]);
        assert_eq!(mul_vec(&m, [1.0, 2.0, 3.0]), [1.0, 2.0, -3.0]);
        assert!((determinant(&m) + 1.0).abs() < 1e-12);
    }
}
