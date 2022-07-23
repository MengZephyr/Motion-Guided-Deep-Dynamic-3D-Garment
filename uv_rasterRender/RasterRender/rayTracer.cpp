#include "rayTracer.h"
#include "mesh.h"
#include <limits.h>

#include <xmmintrin.h>

RayIntersection::RayIntersection()
{
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

	//device = rtcNewDevice("verbose=1");
	device = rtcNewDevice(NULL);
	scene = rtcNewScene(device);
	pShapeMesh = 0;
	vertices = 0;
	triangles = 0;
	ifSceneComm = false;
}

RayIntersection::~RayIntersection()
{
	rtcReleaseScene(scene);
	rtcReleaseDevice(device);
}

unsigned int RayIntersection::addObj(R_Mesh* pMesh)
{
	int fLen = pMesh->numF();
	int vLen = pMesh->numV();

	RTCGeometry mesh = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

	vertices = (EMB_Vertex*)rtcSetNewGeometryBuffer(mesh, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(EMB_Vertex), vLen);
	for (int vi = 0; vi < vLen; vi++)
	{
		cv::Vec3f pp = pMesh->verts[vi];
		vertices[vi].x = pp[0];
		vertices[vi].y = pp[1];
		vertices[vi].z = pp[2];
	}
	triangles = (EMB_Triangle*)rtcSetNewGeometryBuffer(mesh, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(EMB_Triangle), fLen);
	for (int fi = 0; fi < fLen; fi++)
	{
		triangles[fi].v0 = pMesh->faceInds[fi][0];
		triangles[fi].v1 = pMesh->faceInds[fi][1];
		triangles[fi].v2 = pMesh->faceInds[fi][2];
	}
	rtcCommitGeometry(mesh);
	unsigned int geomID = rtcAttachGeometry(scene, mesh);
	rtcReleaseGeometry(mesh);

	return geomID;
}

unsigned int RayIntersection::addObj(std::vector<cv::Vec3f> verts, std::vector<cv::Vec3i> fInds)
{
	int fLen = fInds.size();
	int vLen = verts.size();

	RTCGeometry mesh = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

	vertices = (EMB_Vertex*)rtcSetNewGeometryBuffer(mesh, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(EMB_Vertex), vLen);
	for (int vi = 0; vi < vLen; vi++)
	{
		cv::Vec3f pp = verts[vi];
		vertices[vi].x = pp[0];
		vertices[vi].y = pp[1];
		vertices[vi].z = pp[2];
	}

	triangles = (EMB_Triangle*)rtcSetNewGeometryBuffer(mesh, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(EMB_Triangle), fLen);
	for (int fi = 0; fi < fLen; fi++)
	{
		triangles[fi].v0 = fInds[fi][0];
		triangles[fi].v1 = fInds[fi][1];
		triangles[fi].v2 = fInds[fi][2];
	}

	rtcCommitGeometry(mesh);
	unsigned int geomID = rtcAttachGeometry(scene, mesh);
	rtcReleaseGeometry(mesh);
	printf("geoID: %d\n", geomID);

	return geomID;
}

RTCHit RayIntersection::rayIntersection(cv::Vec3f& ori, cv::Vec3f& dir)
{
	if (!ifSceneComm)
		setSceneCommit();

	dir = normalize(dir);
	RTCRay ray;
	ray.org_x = ori[0];
	ray.org_y = ori[1];
	ray.org_z = ori[2];
	ray.dir_x = dir[0];
	ray.dir_y = dir[1];
	ray.dir_z = dir[2];
	ray.tnear = 0.0f;
	ray.tfar = FLT_MAX;
	ray.mask = -1;
	ray.time = 0;
	RTCRayHit rayHit;
	rayHit.ray = ray;
	rayHit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
	rayHit.hit.primID = RTC_INVALID_GEOMETRY_ID;

	RTCIntersectContext context;
	rtcInitIntersectContext(&context);
	rtcIntersect1(scene, &context, &rayHit);
	return rayHit.hit;
}

cv::Vec3f RayIntersection::interPos(RTCHit& h)
{
	int fID = h.primID;
	cv::Vec3f pp(0., 0., 0.);
	pp[0] = (1. - h.u - h.v) * vertices[triangles[fID].v0].x + h.u * vertices[triangles[fID].v1].x + h.v * vertices[triangles[fID].v2].x;
	pp[1] = (1. - h.u - h.v) * vertices[triangles[fID].v0].y + h.u * vertices[triangles[fID].v1].y + h.v * vertices[triangles[fID].v2].y;
	pp[2] = (1. - h.u - h.v) * vertices[triangles[fID].v0].z + h.u * vertices[triangles[fID].v1].z + h.v * vertices[triangles[fID].v2].z;
	return pp;
}

void RayIntersection::setSceneCommit()
{
	rtcCommitScene(scene);
	ifSceneComm = true;
}
