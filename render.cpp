// ====================================================================
// Simple Render System
// Author:陈保进
// Date：2016.7.5
//=====================================================================



#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <windows.h>
#include <tchar.h>

typedef unsigned int UINT32;

//=====================================================================
// 3D数学库：包括向量计算、矩阵变换
//=====================================================================
typedef struct {
	float m[4][4];
} matrix_4x4;

typedef struct { 
	float x, y, z, w;
} vector_1x4;

int checkValue(int x, int min, int max) {
	return (x < min) ? min : ((x > max) ? max : x); 
}

// 计算插值：t 为 [0, 1] 之间的数值
float interp(float x1, float x2, float t) {
	return x1 + (x2 - x1) * t; 
}

// | v |
float vectorLength(const vector_1x4 * v) {
	float sq = v->x * v->x + v->y * v->y + v->z * v->z;
	return static_cast<float>(sqrt(sq));
}

// z = x + y
void vectorAdd(vector_1x4 * z, const vector_1x4 * x, const vector_1x4 * y) {
	z->x = x->x + y->x;
	z->y = x->y + y->y;
	z->z = x->z + y->z;
	z->w = 1.0;
}

// z = x - y
void vectorSubduct(vector_1x4 * z, const vector_1x4 * x, const vector_1x4 * y) {
	z->x = x->x - y->x;
	z->y = x->y - y->y;
	z->z = x->z - y->z;
	z->w = 1.0;
}

// 矢量点乘
float vectorDotProduct(const vector_1x4 * x, const vector_1x4 * y) {
	return x->x * y->x + x->y * y->y + x->z * y->z;
}

// 矢量叉乘
void vectorCrossProduct(vector_1x4 * z, const vector_1x4 * x, const vector_1x4 * y) {
	float m1, m2, m3;
	m1 = x->y * y->z - x->z * y->y;
	m2 = x->z * y->x - x->x * y->z;
	m3 = x->x * y->y - x->y * y->x;
	z->x = m1;
	z->y = m2;
	z->z = m3;
	z->w = 1.0f;
}

// 矢量插值，t取值 [0, 1]
void vectorInterp(vector_1x4 * z, const vector_1x4 * x1, const vector_1x4 * x2, float t) {
	z->x = interp(x1->x, x2->x, t);
	z->y = interp(x1->y, x2->y, t);
	z->z = interp(x1->z, x2->z, t);
	z->w = 1.0f;
}

// 矢量归一化
void vectorNormalize(vector_1x4 * v) {
	auto length = vectorLength(v);
	if (length != 0.0f) {
		auto inv = 1.0f / length;
		v->x *= inv; 
		v->y *= inv;
		v->z *= inv;
	}
}

// c = a + b
void matrixAdd(matrix_4x4 * c, const matrix_4x4 * a, const matrix_4x4 * b) {
	auto i = 0, j = 0;
	for (; i < 4; i++) {
		for (; j < 4; j++)
			c->m[i][j] = a->m[i][j] + b->m[i][j];
	}
}

// c = a - b
void matrixSub(matrix_4x4 * c, const matrix_4x4 * a, const matrix_4x4 * b) {
	auto i = 0, j = 0;
	for (; i < 4; i++) {
		for (; j < 4; j++)
			c->m[i][j] = a->m[i][j] - b->m[i][j];
	}
}

// c = a * b
void matrixMul(matrix_4x4 * c, const matrix_4x4 * a, const matrix_4x4 * b) {
	matrix_4x4 z;
	for (auto i = 0; i < 4; i++) {
		for (auto j = 0; j < 4; j++) {
			z.m[j][i] = (a->m[j][0] * b->m[0][i]) +
						(a->m[j][1] * b->m[1][i]) +
						(a->m[j][2] * b->m[2][i]) +
						(a->m[j][3] * b->m[3][i]);
		}
	}
	c[0] = z;
}

// c = a * f
void matrixScale(matrix_4x4 * c, const matrix_4x4 * a, const float f) {
	for (auto i = 0; i < 4; i++) {
		for (auto j = 0; j < 4; j++) 
			c->m[i][j] = a->m[i][j] * f;
	}
}

// y = x * m 一维矩阵*方阵
void matrixApply(vector_1x4 * y, const vector_1x4 * x, const matrix_4x4 * m) {
	auto X = x->x, Y = x->y, Z = x->z, W = x->w;
	y->x = X * m->m[0][0] + Y * m->m[1][0] + Z * m->m[2][0] + W * m->m[3][0];  
	y->y = X * m->m[0][1] + Y * m->m[1][1] + Z * m->m[2][1] + W * m->m[3][1];
	y->z = X * m->m[0][2] + Y * m->m[1][2] + Z * m->m[2][2] + W * m->m[3][2];
	y->w = X * m->m[0][3] + Y * m->m[1][3] + Z * m->m[2][3] + W * m->m[3][3];
}

// m = I
void matrixSetIdentity(matrix_4x4 * m) {
	m->m[0][0] = m->m[1][1] = m->m[2][2] = m->m[3][3] = 1.0f; 
	m->m[0][1] = m->m[0][2] = m->m[0][3] = 0.0f;
	m->m[1][0] = m->m[1][2] = m->m[1][3] = 0.0f;
	m->m[2][0] = m->m[2][1] = m->m[2][3] = 0.0f;
	m->m[3][0] = m->m[3][1] = m->m[3][2] = 0.0f;
}

// m = 0
void matrixSetZero(matrix_4x4 * m) {
	m->m[0][0] = m->m[0][1] = m->m[0][2] = m->m[0][3] = 0.0f;
	m->m[1][0] = m->m[1][1] = m->m[1][2] = m->m[1][3] = 0.0f;
	m->m[2][0] = m->m[2][1] = m->m[2][2] = m->m[2][3] = 0.0f;
	m->m[3][0] = m->m[3][1] = m->m[3][2] = m->m[3][3] = 0.0f;
}

// 平移变换
void matrixSetTranslate(matrix_4x4 * m, const float x, const float y, const float z) {
	matrixSetIdentity(m);
	m->m[3][0] = x;
	m->m[3][1] = y;
	m->m[3][2] = z; 
}

// 缩放变换
void matrixSetScale(matrix_4x4 * m, const float x, const float y, const float z) {
	matrixSetIdentity(m);
	m->m[0][0] = x;
	m->m[1][1] = y;
	m->m[2][2] = z;
}

// 旋转变换
void matrixSetRotate(matrix_4x4 * m, float x, float y, float z, const float theta) {
	auto qsin = static_cast<float>(sin(theta * 0.5f));
	auto qcos = static_cast<float>(cos(theta * 0.5f));
	vector_1x4 vec = { x, y, z, 1.0f };
	auto w = qcos;
	vectorNormalize(&vec);
	x = vec.x * qsin;
	y = vec.y * qsin;
	z = vec.z * qsin;
	m->m[0][0] = 1 - 2 * y * y - 2 * z * z;
	m->m[1][0] = 2 * x * y - 2 * w * z;
	m->m[2][0] = 2 * x * z + 2 * w * y;
	m->m[0][1] = 2 * x * y + 2 * w * z;
	m->m[1][1] = 1 - 2 * x * x - 2 * z * z;
	m->m[2][1] = 2 * y * z - 2 * w * x;
	m->m[0][2] = 2 * x * z - 2 * w * y;
	m->m[1][2] = 2 * y * z + 2 * w * x;
	m->m[2][2] = 1 - 2 * x * x - 2 * y * y;
	m->m[0][3] = m->m[1][3] = m->m[2][3] = 0.0f;
	m->m[3][0] = m->m[3][1] = m->m[3][2] = 0.0f;	
	m->m[3][3] = 1.0f;
}

// 设置摄像机
void matrixSetLookat(matrix_4x4 * m, const vector_1x4 * eye, const vector_1x4 * at, const vector_1x4 * up) {
	vector_1x4 axisX, axisY, axisZ;

	vectorSubduct(&axisZ, at, eye);
	vectorNormalize(&axisZ);
	vectorCrossProduct(&axisX, up, &axisZ);
	vectorNormalize(&axisX);
	vectorCrossProduct(&axisY, &axisZ, &axisX);

	m->m[0][0] = axisX.x;
	m->m[1][0] = axisX.y;
	m->m[2][0] = axisX.z;
	m->m[3][0] = -vectorDotProduct(&axisX, eye);

	m->m[0][1] = axisY.x;
	m->m[1][1] = axisY.y;
	m->m[2][1] = axisY.z;
	m->m[3][1] = -vectorDotProduct(&axisY, eye);

	m->m[0][2] = axisZ.x;
	m->m[1][2] = axisZ.y;
	m->m[2][2] = axisZ.z;
	m->m[3][2] = -vectorDotProduct(&axisZ, eye);
	
	m->m[0][3] = m->m[1][3] = m->m[2][3] = 0.0f;
	m->m[3][3] = 1.0f;
}

// D3DXMatrixPerspectiveFovLH 设置投影矩阵
void matrixSetPerspective(matrix_4x4 * m, const float fovy, const float aspect, const float zn, const float zf) {
	auto fax = 1.0f / static_cast<float>(tan(fovy * 0.5f));
	matrixSetZero(m);
	m->m[0][0] = static_cast<float>(fax / aspect);
	m->m[1][1] = static_cast<float>(fax);
	m->m[2][2] = zf / (zf - zn);
	m->m[3][2] = - zn * zf / (zf - zn);
	m->m[2][3] = 1;
}


//=====================================================================
// 坐标变换
//=====================================================================
typedef struct tagTransform { 
	matrix_4x4 world;         // 世界坐标变换
	matrix_4x4 view;          // 摄影机坐标变换
	matrix_4x4 projection;    // 投影变换
	matrix_4x4 transform;     // transform = world * view * projection
	float w, h;             // 屏幕大小
}transform_t;


// 矩阵更新，计算 transform = world * view * projection
void transformUpdate(transform_t * ts) {
	matrix_4x4 m;
	matrixMul(&m, &ts->world, &ts->view);
	matrixMul(&ts->transform, &m, &ts->projection);
}

// 初始化，设置屏幕长宽
void transformInit(transform_t * ts, const int width, const int height) {
	auto aspect = static_cast<float>(width) / static_cast<float>(height);
	matrixSetIdentity(&ts->world);
	matrixSetIdentity(&ts->view);
	matrixSetPerspective(&ts->projection, 3.1415926f * 0.5f, aspect, 1.0f, 500.0f);
	ts->w = static_cast<float>(width);
	ts->h = static_cast<float>(height);
	transformUpdate(ts);
}

// 将 x 投影 
void transformApply(const transform_t * ts, vector_1x4 * y, const vector_1x4 * x) {
	matrixApply(y, x, &ts->transform);
}

// 检查齐次坐标同 cvv 的边界用于视锥裁剪
int transformCheckCVV(const vector_1x4 * v) {
	auto w = v->w;
	auto check = 0;
	if (v->z < 0.0f) check |= 1;
	if (v->z >  w) check |= 2;
	if (v->x < -w) check |= 4;
	if (v->x >  w) check |= 8;
	if (v->y < -w) check |= 16;
	if (v->y >  w) check |= 32;
	return check;
}

// 归一化，得到屏幕坐标
void transformHomogenize(const transform_t * ts, vector_1x4 * y, const vector_1x4 * x) {
	auto rhw = 1.0f / x->w;
	y->x = (x->x * rhw + 1.0f) * ts->w * 0.5f;
	y->y = (1.0f - x->y * rhw) * ts->h * 0.5f;
	y->z = x->z * rhw;
	y->w = 1.0f;
}


//=====================================================================
// 几何计算：顶点、扫描线、边缘、矩形、步长计算
//=====================================================================
typedef struct tagColor { float r, g, b; } color_t;
typedef struct tagTexcoord { float u, v; } texcoord_t;
typedef struct tagVertex { vector_1x4 pos; texcoord_t tc; color_t color; float rhw; } vertex_t;
typedef struct tagEdge { vertex_t v, v1, v2; } edge_t;
typedef struct tagTrapezoid { float top, bottom; edge_t left, right; } trapezoid_t;
typedef struct tagScanline { vertex_t v, step; int x, y, w; } scanline_t;


void vertexRhwInit(vertex_t * v) {
	auto rhw = 1.0f / v->pos.w;
	v->rhw = rhw;
	v->tc.u *= rhw;
	v->tc.v *= rhw;
	v->color.r *= rhw;
	v->color.g *= rhw;
	v->color.b *= rhw;
}

// 顶点插值
void vertexInterp(vertex_t * y, const vertex_t * x1, const vertex_t * x2, float t) {
	vectorInterp(&y->pos, &x1->pos, &x2->pos, t);
	y->tc.u = interp(x1->tc.u, x2->tc.u, t);
	y->tc.v = interp(x1->tc.v, x2->tc.v, t);
	y->color.r = interp(x1->color.r, x2->color.r, t);
	y->color.g = interp(x1->color.g, x2->color.g, t);
	y->color.b = interp(x1->color.b, x2->color.b, t);
	y->rhw = interp(x1->rhw, x2->rhw, t);
}

void vertexDivision(vertex_t * y, const vertex_t * x1, const vertex_t * x2, float w) {
	auto inv = 1.0f / w;
	y->pos.x = (x2->pos.x - x1->pos.x) * inv;
	y->pos.y = (x2->pos.y - x1->pos.y) * inv;
	y->pos.z = (x2->pos.z - x1->pos.z) * inv;
	y->pos.w = (x2->pos.w - x1->pos.w) * inv;
	y->tc.u = (x2->tc.u - x1->tc.u) * inv;
	y->tc.v = (x2->tc.v - x1->tc.v) * inv;
	y->color.r = (x2->color.r - x1->color.r) * inv;
	y->color.g = (x2->color.g - x1->color.g) * inv;
	y->color.b = (x2->color.b - x1->color.b) * inv;
	y->rhw = (x2->rhw - x1->rhw) * inv;
}

void vertexAdd(vertex_t * y, const vertex_t * x) {
	y->pos.x += x->pos.x;
	y->pos.y += x->pos.y;
	y->pos.z += x->pos.z;
	y->pos.w += x->pos.w;
	y->rhw += x->rhw;
	y->tc.u += x->tc.u;
	y->tc.v += x->tc.v;
	y->color.r += x->color.r;
	y->color.g += x->color.g;
	y->color.b += x->color.b;
}

// 根据三角形生成并且返回合法梯形的数量
int trapezoidInitTriangle(trapezoid_t * trap, const vertex_t * p1, const vertex_t * p2, const vertex_t * p3) {
	const vertex_t * p;
	float k, x;

	if (p1->pos.y > p2->pos.y) 
		p = p1, p1 = p2, p2 = p;

	if (p1->pos.y > p3->pos.y) 
		p = p1, p1 = p3, p3 = p;

	if (p2->pos.y > p3->pos.y) 
		p = p2, p2 = p3, p3 = p;

	if (p1->pos.y == p2->pos.y && p1->pos.y == p3->pos.y)
		return 0;

	if (p1->pos.x == p2->pos.x && p1->pos.x == p3->pos.x)
		return 0;

	if (p1->pos.y == p2->pos.y) {	// triangle down
		if (p1->pos.x > p2->pos.x) p = p1, p1 = p2, p2 = p;
		trap[0].top = p1->pos.y;
		trap[0].bottom = p3->pos.y;
		trap[0].left.v1 = *p1;
		trap[0].left.v2 = *p3;
		trap[0].right.v1 = *p2;
		trap[0].right.v2 = *p3;
		return (trap[0].top < trap[0].bottom)? 1 : 0;
	}

	if (p2->pos.y == p3->pos.y) {	// triangle up
		if (p2->pos.x > p3->pos.x) p = p2, p2 = p3, p3 = p;
		trap[0].top = p1->pos.y;
		trap[0].bottom = p3->pos.y;
		trap[0].left.v1 = *p1;
		trap[0].left.v2 = *p2;
		trap[0].right.v1 = *p1;
		trap[0].right.v2 = *p3;
		return (trap[0].top < trap[0].bottom)? 1 : 0;
	}

	trap[0].top = p1->pos.y;
	trap[0].bottom = p2->pos.y;
	trap[1].top = p2->pos.y;
	trap[1].bottom = p3->pos.y;

	k = (p3->pos.y - p1->pos.y) / (p2->pos.y - p1->pos.y);
	x = p1->pos.x + (p2->pos.x - p1->pos.x) * k;

	if (x <= p3->pos.x) {		// triangle left
		trap[0].left.v1 = *p1;
		trap[0].left.v2 = *p2;
		trap[0].right.v1 = *p1;
		trap[0].right.v2 = *p3;
		trap[1].left.v1 = *p2;
		trap[1].left.v2 = *p3;
		trap[1].right.v1 = *p1;
		trap[1].right.v2 = *p3;
	}	else {					// triangle right
		trap[0].left.v1 = *p1;
		trap[0].left.v2 = *p3;
		trap[0].right.v1 = *p1;
		trap[0].right.v2 = *p2;
		trap[1].left.v1 = *p1;
		trap[1].left.v2 = *p3;
		trap[1].right.v1 = *p2;
		trap[1].right.v2 = *p3;
	}

	return 2;
}

// 按照 Y 坐标计算出左右两条边纵坐标等于 Y 的顶点
void trapezoidEdgeInterp(trapezoid_t * trap, float y) {
	float s1 = trap->left.v2.pos.y - trap->left.v1.pos.y;
	float s2 = trap->right.v2.pos.y - trap->right.v1.pos.y;
	float t1 = (y - trap->left.v1.pos.y) / s1;
	float t2 = (y - trap->right.v1.pos.y) / s2;
	vertexInterp(&trap->left.v, &trap->left.v1, &trap->left.v2, t1);
	vertexInterp(&trap->right.v, &trap->right.v1, &trap->right.v2, t2);
}

// 根据左右两边的端点，初始化计算出扫描线的起点和步长
void trapezoidInitScanLine(const trapezoid_t * trap, scanline_t * scanline, int y) {
	float width = trap->right.v.pos.x - trap->left.v.pos.x;
	scanline->x = (int)(trap->left.v.pos.x + 0.5f);
	scanline->w = (int)(trap->right.v.pos.x + 0.5f) - scanline->x;
	scanline->y = y;
	scanline->v = trap->left.v;
	if (trap->left.v.pos.x >= trap->right.v.pos.x) 
		scanline->w = 0;

	vertexDivision(&scanline->step, &trap->left.v, &trap->right.v, width);
}


//=====================================================================
// 渲染设备
//=====================================================================
typedef struct tagDevice {
	transform_t transform;      // 坐标变换器
	int width;                  // 窗口宽度
	int height;                 // 窗口高度
	UINT32 **framebuffer;      // 像素缓存：framebuffer[y] 代表第 y行
	float **zbuffer;            // 深度缓存：zbuffer[y] 为第 y行指针
	UINT32 **texture;          // 纹理：同样是每行索引
	int texWidth;              // 纹理宽度
	int texHeight;             // 纹理高度
	float maxU;                // 纹理最大宽度：tex_width - 1
	float maxV;                // 纹理最大高度：tex_height - 1
	int renderState;           // 渲染状态
	UINT32 background;         // 背景颜色
	UINT32 foreground;         // 线框颜色
}	device_t;

enum 
{
	RENDER_STATE_WIREFRAME = 1,		// 渲染线框
	RENDER_STATE_TEXTURE = 2,		// 渲染纹理
	RENDER_STATE_COLOR = 4			// 渲染颜色
};

// 设备初始化，fb为外部帧缓存，非 NULL 将引用外部帧缓存（每行 4字节对齐）
void deviceInit(device_t * device, const int width, const int height, void * fb) {
	int need = sizeof(void*) * (height * 2 + 1024) + width * height * 8;
	char * ptr = new char[need + 64];
	char * framebuf, * zbuf;
	int j;
	assert(ptr);
	device->framebuffer = (UINT32**)ptr;
	device->zbuffer = (float**)(ptr + sizeof(void*) * height);
	ptr += sizeof(void*) * height * 2;
	device->texture = (UINT32**)ptr;
	ptr += sizeof(void*) * 1024;
	framebuf = (char*)ptr;
	zbuf = (char*)ptr + width * height * 4;
	ptr += width * height * 8;
	if (fb != NULL) framebuf = (char*)fb;
	for (j = 0; j < height; j++) {
		device->framebuffer[j] = (UINT32*)(framebuf + width * 4 * j);
		device->zbuffer[j] = (float*)(zbuf + width * 4 * j);
	}
	device->texture[0] = (UINT32*)ptr;
	device->texture[1] = (UINT32*)(ptr + 16);
	memset(device->texture[0], 0, 64);
	device->texWidth = 2;
	device->texHeight = 2;
	device->maxU = 1.0f;
	device->maxV = 1.0f;
	device->width = width;
	device->height = height;
	device->background = 0xc0c0c0;
	device->foreground = 0;
	transformInit(&device->transform, width, height);
	device->renderState = RENDER_STATE_WIREFRAME;
}

// 删除设备
void deviceDestroy(device_t * device) {
	if (device->framebuffer) 
		delete device->framebuffer;
	device->framebuffer = NULL;
	device->zbuffer = NULL;
	device->texture = NULL;
}

// 设置当前纹理
void deviceSetTexture(device_t * device, void * bits, long pitch, int w, int h) {
	char *ptr = (char*)bits;
	int j;
	assert(w <= 1024 && h <= 1024);
	for (j = 0; j < h; ptr += pitch, j++) 	// 重新计算每行纹理的指针
		device->texture[j] = (UINT32*)ptr;
	device->texWidth = w;
	device->texHeight = h;
	device->maxU = (float)(w - 1);
	device->maxV = (float)(h - 1);
}

// 清空 framebuffer 和 zbuffer
void deviceClear(device_t * device, int mode) {
	int y, x, height = device->height;
	for (y = 0; y < device->height; y++) {
		UINT32 *dst = device->framebuffer[y];
		UINT32 cc = (height - 1 - y) * 230 / (height - 1);
		cc = (cc << 16) | (cc << 8) | cc;
		if (mode == 0) cc = device->background;
		for (x = device->width; x > 0; dst++, x--) dst[0] = cc;
	}
	for (y = 0; y < device->height; y++) {
		float *dst = device->zbuffer[y];
		for (x = device->width; x > 0; dst++, x--) dst[0] = 0.0f;
	}
}

// 画点
void drawPixel(device_t * device, int x, int y, UINT32 color) {
	if (((UINT32)x) < (UINT32)device->width && ((UINT32)y) < (UINT32)device->height) {
		device->framebuffer[y][x] = color;
	}
}

// 绘制线段
void drawLine(device_t * device, int x1, int y1, int x2, int y2, UINT32 c) {
	int x, y, rem = 0;
	if (x1 == x2 && y1 == y2) {
		drawPixel(device, x1, y1, c);
	}	else if (x1 == x2) {
		int inc = (y1 <= y2)? 1 : -1;
		for (y = y1; y != y2; y += inc) drawPixel(device, x1, y, c);
		drawPixel(device, x2, y2, c);
	}	else if (y1 == y2) {
		int inc = (x1 <= x2)? 1 : -1;
		for (x = x1; x != x2; x += inc) drawPixel(device, x, y1, c);
		drawPixel(device, x2, y2, c);
	}	else {
		int dx = (x1 < x2)? x2 - x1 : x1 - x2;
		int dy = (y1 < y2)? y2 - y1 : y1 - y2;
		if (dx >= dy) {
			if (x2 < x1) x = x1, y = y1, x1 = x2, y1 = y2, x2 = x, y2 = y;
			for (x = x1, y = y1; x <= x2; x++) {
				drawPixel(device, x, y, c);
				rem += dy;
				if (rem >= dx) {
					rem -= dx;
					y += (y2 >= y1)? 1 : -1;
					drawPixel(device, x, y, c);
				}
			}
			drawPixel(device, x2, y2, c);
		}	else {
			if (y2 < y1) x = x1, y = y1, x1 = x2, y1 = y2, x2 = x, y2 = y;
			for (x = x1, y = y1; y <= y2; y++) {
				drawPixel(device, x, y, c);
				rem += dx;
				if (rem >= dy) {
					rem -= dy;
					x += (x2 >= x1)? 1 : -1;
					drawPixel(device, x, y, c);
				}
			}
			drawPixel(device, x2, y2, c);
		}
	}
}

// 根据坐标读取纹理
UINT32 textureRead(const device_t * device, float u, float v) {
	int x, y;
	u = u * device->maxU;
	v = v * device->maxV;
	x = (int)(u + 0.5f);
	y = (int)(v + 0.5f);
	x = checkValue(x, 0, device->texWidth - 1);
	y = checkValue(y, 0, device->texHeight - 1);
	return device->texture[y][x];
}


//=====================================================================
// 渲染实现
//=====================================================================

// 绘制扫描线
void drawScanLine(device_t * device, scanline_t * scanline) {
	UINT32 *framebuffer = device->framebuffer[scanline->y];
	float *zbuffer = device->zbuffer[scanline->y];
	int x = scanline->x;
	int w = scanline->w;
	int width = device->width;
	int render_state = device->renderState;
	for (; w > 0; x++, w--) {
		if (x >= 0 && x < width) {
			float rhw = scanline->v.rhw;
			if (rhw >= zbuffer[x]) {	
				float w = 1.0f / rhw;
				zbuffer[x] = rhw;
				if (render_state & RENDER_STATE_COLOR) {
					float r = scanline->v.color.r * w;
					float g = scanline->v.color.g * w;
					float b = scanline->v.color.b * w;
					int R = (int)(r * 255.0f);
					int G = (int)(g * 255.0f);
					int B = (int)(b * 255.0f);
					R = checkValue(R, 0, 255);
					G = checkValue(G, 0, 255);
					B = checkValue(B, 0, 255);
					framebuffer[x] = (R << 16) | (G << 8) | (B);
				}
				if (render_state & RENDER_STATE_TEXTURE) {
					float u = scanline->v.tc.u * w;
					float v = scanline->v.tc.v * w;
					UINT32 cc = textureRead(device, u, v);
					framebuffer[x] = cc;
				}
			}
		}
		vertexAdd(&scanline->v, &scanline->step);
		if (x >= width) break;
	}
}

// 绘制梯形
void drawTrapezoid(device_t * device, trapezoid_t * trap) {
	scanline_t scanline;
	int j, top, bottom;
	top = (int)(trap->top + 0.5f);
	bottom = (int)(trap->bottom + 0.5f);
	for (j = top; j < bottom; j++) {
		if (j >= 0 && j < device->height) {
			trapezoidEdgeInterp(trap, (float)j + 0.5f);
			trapezoidInitScanLine(trap, &scanline, j);
			drawScanLine(device, &scanline);
		}
		if (j >= device->height) break;
	}
}

// 根据 render_state 绘制原始三角形
void drawTriangle(device_t * device, const vertex_t * v1, const vertex_t * v2, const vertex_t * v3) {
	vector_1x4 p1, p2, p3, c1, c2, c3;
	int render_state = device->renderState;

	// 按照 Transform 变化
	transformApply(&device->transform, &c1, &v1->pos);
	transformApply(&device->transform, &c2, &v2->pos);
	transformApply(&device->transform, &c3, &v3->pos);

	// 裁剪，注意此处可以完善为具体判断几个点在 cvv内以及同cvv相交平面的坐标比例
	// 进行进一步精细裁剪，将一个分解为几个完全处在 cvv内的三角形
	if (transformCheckCVV(&c1) != 0) return;
	if (transformCheckCVV(&c2) != 0) return;
	if (transformCheckCVV(&c3) != 0) return;

	// 归一化
	transformHomogenize(&device->transform, &p1, &c1);
	transformHomogenize(&device->transform, &p2, &c2);
	transformHomogenize(&device->transform, &p3, &c3);

	// 纹理或者色彩绘制
	if (render_state & (RENDER_STATE_TEXTURE | RENDER_STATE_COLOR)) {
		vertex_t t1 = *v1, t2 = *v2, t3 = *v3;
		trapezoid_t traps[2];
		int n;

		t1.pos = p1; 
		t2.pos = p2;
		t3.pos = p3;
		t1.pos.w = c1.w;
		t2.pos.w = c2.w;
		t3.pos.w = c3.w;
		
		vertexRhwInit(&t1);	// 初始化 w
		vertexRhwInit(&t2);	// 初始化 w
		vertexRhwInit(&t3);	// 初始化 w
		
		// 拆分三角形为0-2个梯形，并且返回可用梯形数量
		n = trapezoidInitTriangle(traps, &t1, &t2, &t3);

		if (n >= 1) drawTrapezoid(device, &traps[0]);
		if (n >= 2) drawTrapezoid(device, &traps[1]);
	}

	if (render_state & RENDER_STATE_WIREFRAME) {		// 线框绘制
		drawLine(device, (int)p1.x, (int)p1.y, (int)p2.x, (int)p2.y, device->foreground);
		drawLine(device, (int)p1.x, (int)p1.y, (int)p3.x, (int)p3.y, device->foreground);
		drawLine(device, (int)p3.x, (int)p3.y, (int)p2.x, (int)p2.y, device->foreground);
	}
}


//=====================================================================
// Win32 窗口及图形绘制：为 device 提供一个 DibSection 的 FB
//=====================================================================
int screen_w, screen_h, screen_exit = 0;
int screen_mx = 0, screen_my = 0, screen_mb = 0;
int screen_keys[512];	// 当前键盘按下状态
static HWND screen_handle = NULL;		// 主窗口 HWND
static HDC screen_dc = NULL;			// 配套的 HDC
static HBITMAP screen_hb = NULL;		// DIB
static HBITMAP screen_ob = NULL;		// 老的 BITMAP
unsigned char *screen_fb = NULL;		// frame buffer
long screen_pitch = 0;

int screenInit(int w, int h, const TCHAR * title);	// 屏幕初始化
int screenClose(void);								// 关闭屏幕
void screenMsgDispatch(void);						// 处理消息
void screenUpdate(void);							// 显示 FrameBuffer

// win32 event handler
static LRESULT screenEvents(HWND, UINT, WPARAM, LPARAM);	

#ifdef _MSC_VER
#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "user32.lib")
#endif

// 初始化窗口并设置标题
int screenInit(int w, int h, const TCHAR *title) {
	WNDCLASS wc = { CS_BYTEALIGNCLIENT, (WNDPROC)screenEvents, 0, 0, 0, 
		NULL, NULL, NULL, NULL, _T("SCREEN3.1415926") };
	BITMAPINFO bi = { { sizeof(BITMAPINFOHEADER), w, -h, 1, 32, BI_RGB, 
		w * h * 4, 0, 0, 0, 0 }  };
	RECT rect = { 0, 0, w, h };
	int wx, wy, sx, sy;
	LPVOID ptr;
	HDC hDC;

	screenClose();

	wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	wc.hInstance = GetModuleHandle(NULL);
	wc.hCursor = LoadCursor(NULL, IDC_ARROW);
	if (!RegisterClass(&wc)) return -1;

	screen_handle = CreateWindow(_T("SCREEN3.1415926"), title,
		WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX,
		0, 0, 0, 0, NULL, NULL, wc.hInstance, NULL);
	if (screen_handle == NULL) return -2;

	screen_exit = 0;
	hDC = GetDC(screen_handle);
	screen_dc = CreateCompatibleDC(hDC);
	ReleaseDC(screen_handle, hDC);

	screen_hb = CreateDIBSection(screen_dc, &bi, DIB_RGB_COLORS, &ptr, 0, 0);
	if (screen_hb == NULL) return -3;

	screen_ob = (HBITMAP)SelectObject(screen_dc, screen_hb);
	screen_fb = (unsigned char*)ptr;
	screen_w = w;
	screen_h = h;
	screen_pitch = w * 4;
	
	AdjustWindowRect(&rect, GetWindowLong(screen_handle, GWL_STYLE), 0);
	wx = rect.right - rect.left;
	wy = rect.bottom - rect.top;
	sx = (GetSystemMetrics(SM_CXSCREEN) - wx) / 2;
	sy = (GetSystemMetrics(SM_CYSCREEN) - wy) / 2;
	if (sy < 0) sy = 0;
	SetWindowPos(screen_handle, NULL, sx, sy, wx, wy, (SWP_NOCOPYBITS | SWP_NOZORDER | SWP_SHOWWINDOW));
	SetForegroundWindow(screen_handle);

	ShowWindow(screen_handle, SW_NORMAL);
	screenMsgDispatch();

	memset(screen_keys, 0, sizeof(int) * 512);
	memset(screen_fb, 0, w * h * 4);

	return 0;
}

int screenClose(void) {
	if (screen_dc) {
		if (screen_ob) { 
			SelectObject(screen_dc, screen_ob); 
			screen_ob = NULL; 
		}
		DeleteDC(screen_dc);
		screen_dc = NULL;
	}
	if (screen_hb) { 
		DeleteObject(screen_hb); 
		screen_hb = NULL; 
	}
	if (screen_handle) { 
		CloseWindow(screen_handle); 
		screen_handle = NULL; 
	}
	return 0;
}

static LRESULT screenEvents(HWND hWnd, UINT msg, 
	WPARAM wParam, LPARAM lParam) {
	switch (msg) {
	case WM_CLOSE: 
		screen_exit = 1; break;
	case WM_KEYDOWN: 
		screen_keys[wParam & 511] = 1; break;
	case WM_MOUSEWHEEL:
	{
		int keys = LOWORD(wParam);
		int delta = HIWORD(wParam);
		screen_keys[wParam & 511] = 1;
		break;
	}
	case WM_KEYUP: 
		screen_keys[wParam & 511] = 0; break;
	default: return DefWindowProc(hWnd, msg, wParam, lParam);
	}
	return 0;
}

void screenMsgDispatch(void) {
	MSG msg;
	while (1) {
		if (!PeekMessage(&msg, NULL, 0, 0, PM_NOREMOVE)) break;
		if (!GetMessage(&msg, NULL, 0, 0)) break;
		DispatchMessage(&msg);
	}
}

void screenUpdate(void) {
	HDC hDC = GetDC(screen_handle);
	BitBlt(hDC, 0, 0, screen_w, screen_h, screen_dc, 0, 0, SRCCOPY);
	ReleaseDC(screen_handle, hDC);
	screenMsgDispatch();
}


//=====================================================================
// 主程序
//=====================================================================
vertex_t mesh[8] = {
	{ {  1, -1,  1, 1 }, { 0, 0 }, { 1.0f, 0.2f, 0.2f }, 1 },
	{ { -1, -1,  1, 1 }, { 0, 1 }, { 0.2f, 1.0f, 0.2f }, 1 },
	{ { -1,  1,  1, 1 }, { 1, 1 }, { 0.2f, 0.2f, 1.0f }, 1 },
	{ {  1,  1,  1, 1 }, { 1, 0 }, { 1.0f, 0.2f, 1.0f }, 1 },
	{ {  1, -1, -1, 1 }, { 0, 0 }, { 1.0f, 1.0f, 0.2f }, 1 },
	{ { -1, -1, -1, 1 }, { 0, 1 }, { 0.2f, 1.0f, 1.0f }, 1 },
	{ { -1,  1, -1, 1 }, { 1, 1 }, { 1.0f, 0.3f, 0.3f }, 1 },
	{ {  1,  1, -1, 1 }, { 1, 0 }, { 0.2f, 1.0f, 0.3f }, 1 },
};

void drawPlane(device_t * device, int a, int b, int c, int d) {
	vertex_t p1 = mesh[a], p2 = mesh[b], p3 = mesh[c], p4 = mesh[d];
	p1.tc.u = 0, p1.tc.v = 0, p2.tc.u = 0, p2.tc.v = 1;
	p3.tc.u = 1, p3.tc.v = 1, p4.tc.u = 1, p4.tc.v = 0;
	drawTriangle(device, &p1, &p2, &p3);
	drawTriangle(device, &p3, &p4, &p1);
}

void drawBox(device_t * device, float theta) {
	matrix_4x4 m;
	matrixSetRotate(&m, -1, -0.5, 1, theta);
	device->transform.world = m;
	transformUpdate(&device->transform);
	drawPlane(device, 0, 1, 2, 3);
	drawPlane(device, 4, 5, 6, 7);
	drawPlane(device, 0, 4, 5, 1);
	drawPlane(device, 1, 5, 6, 2);
	drawPlane(device, 2, 6, 7, 3);
	drawPlane(device, 3, 7, 4, 0);
}

void cameraAtZero(device_t * device, float x, float y, float z) {
	vector_1x4 eye = { x, y, z, 1 }, at = { 0, 0, 0, 1 }, up = { 0, 0, 1, 1 };
	matrixSetLookat(&device->transform.view, &eye, &at, &up);
	transformUpdate(&device->transform);
}

void textureInit(device_t * device) {
	static UINT32 texture[256][256];
	int i, j;
	for (j = 0; j < 256; j++) {
		for (i = 0; i < 256; i++) {
			int x = i / 32, y = j / 32;
			texture[j][i] = ((x + y) & 1)? 0xffffff : 0x3fbcef;
		}
	}
	deviceSetTexture(device, texture, 256 * 4, 256, 256);
}

int main(void)
{
	device_t device;
	int states[] = { RENDER_STATE_TEXTURE, RENDER_STATE_COLOR, RENDER_STATE_WIREFRAME };
	int indicator = 0;
	int kbhit = 0;
	float alpha = 1;
	float pos = 3.5;

	TCHAR *title = _T("3D render demo - ")
		_T("Left/Right: rotation, Up/Down: forward/backward, Space: switch state");

	if (screenInit(800, 600, title)) 
		return -1;

	deviceInit(&device, 800, 600, screen_fb);
	cameraAtZero(&device, 3, 0, 0);

	textureInit(&device);
	device.renderState = RENDER_STATE_TEXTURE;

	while (screen_exit == 0 && screen_keys[VK_ESCAPE] == 0) {
		screenMsgDispatch();
		deviceClear(&device, 1);
		cameraAtZero(&device, pos, 0, 0);
		
		if (screen_keys[VK_UP])
			pos -= 0.01f;
		if (screen_keys[VK_DOWN])
			pos += 0.01f;
		if (screen_keys[VK_LEFT])
			alpha += 0.01f;
		if (screen_keys[VK_RIGHT])
			alpha -= 0.01f;

		if (screen_keys[VK_SPACE]) {
			if (kbhit == 0) {
				kbhit = 1;

				if (++indicator >= 3) 
					indicator = 0;

				device.renderState = states[indicator];
			}
		}	
		else {
			kbhit = 0;
		}

		drawBox(&device, alpha);
		screenUpdate();
		//Sleep(1);
	}

	return 0;
}

