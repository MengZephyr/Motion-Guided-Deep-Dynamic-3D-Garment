#include "DataIO.h"
#include "mesh.h"
#include "rayTracer.h"
//#include <omp.h>

cv::Mat resampleImgs(cv::Mat img, int thumbHeight, int thumbWidth)
{
	int height = img.rows;
	int width = img.cols;
	double xscale = (thumbWidth + 0.0) / double(width);
	double yscale = (thumbHeight + 0.0) / double(height);
	double threshold = 0.5 / (xscale * yscale);
	double yend = 0.0;
	cv::Mat rst = cv::Mat::zeros(thumbHeight, thumbWidth, CV_32FC3);

	for (int f = 0; f < thumbHeight; f++) // y on output
	{
		double ystart = yend;
		yend = (f + 1) / yscale;
		if (yend >= height) yend = height - 0.000001;
		double xend = 0.0;
		for (int g = 0; g < thumbWidth; g++) // x on output
		{
			double xstart = xend;
			xend = (g + 1) / xscale;
			if (xend >= width) xend = width - 0.000001;

			cv::Vec3f sum(0.0, 0.0, 0.0);
			double portions = 0.;

			for (int y = (int)ystart; y <= (int)yend; ++y)
			{
				double yportion = 1.0;
				if (y == (int)ystart) yportion -= ystart - y;
				if (y == (int)yend) yportion -= y + 1 - yend;
				for (int x = (int)xstart; x <= (int)xend; ++x)
				{
					double xportion = 1.0;
					if (x == (int)xstart) xportion -= xstart - x;
					if (x == (int)xend) xportion -= x + 1 - xend;
					sum += img.at<cv::Vec3f>(y, x) * yportion * xportion;
					portions += yportion * xportion;
				}
			}
			rst.at<cv::Vec3f>(f, g) = sum / portions;
		}
	}
	return rst;
}

void genUVMask()
{
	std::string caseName = "walk_75/";
	std::string F_prefix = "C:/newProj/SMPL/SMPL_mixamo/walk/";
	std::string objRoot = F_prefix + caseName;

	int uv_height[1] = { 32 };
	int uv_Width[1] = { 32 };

	string uvName = objRoot + "uv/body_uv.ply";
	R_Mesh uvMesh;
	readPly(uvName, uvMesh.verts, uvMesh.faceInds);
	uvMesh.scaleMesh(uv_height[0], uv_Width[0], 1.);
	RayIntersection myTracer;
	myTracer.addObj(&uvMesh);

	string geoName = objRoot + "uv/body_geo.ply";
	R_Mesh geoMesh;
	readPly(geoName, geoMesh.verts, geoMesh.faceInds);

	std::vector<std::vector<cv::Vec2i>> Pixel_Valid(1);
	std::vector<std::vector<cv::Vec3i>> Pixel_VertID(1);
	std::vector<std::vector<cv::Vec2f>> Pixel_Wei(1);

	cv::Mat rstNMap = cv::Mat::zeros(uv_height[0], uv_Width[0], CV_32FC3);

	for (int y = 0; y < uv_height[0]; y++)
	{
		for (int x = 0; x < uv_Width[0]; x++)
		{
			cv::Vec3f ori(x, uv_height[0] - y, -10.);
			cv::Vec3f dir(0., 0., 1.);
			RTCHit h = myTracer.rayIntersection(ori, dir);
			int fID = h.primID;
			if (fID < 0)
				continue;
			else
			{
				rstNMap.at<cv::Vec3f>(y, x) = cv::Vec3f(1., 1., 1.);
			}
		} // end for x
	} // end for y
	cv::imwrite(objRoot + "uv/uv_Mask_32.png", rstNMap * 255.);
}

void getUVIndexEffi()
{
	std::string caseName = "/";
	std::string F_prefix = "D:/models/DS/Data_walk/mixamo_body/sequence/";
	std::string objRoot = F_prefix + caseName;

	int uv_height[1] = { 256 };
	int uv_Width[1] = { 256 };

	string uvName = objRoot + "skirt_uv/garment_uv.ply";
	R_Mesh uvMesh;
	readPly(uvName, uvMesh.verts, uvMesh.faceInds);
	uvMesh.scaleMesh(uv_height[0], uv_Width[0], 1.);
	
	std::vector<cv::Vec2i> p0(uvMesh.numV());
	std::vector<cv::Vec2i> p1(uvMesh.numV());
	std::vector<cv::Vec2i> p2(uvMesh.numV());
	std::vector<cv::Vec2i> p3(uvMesh.numV());

	std::vector<cv::Vec4f> efi(uvMesh.numV());
	
	for (int vi = 0; vi < uvMesh.numV(); vi++)
	{
		cv::Vec3f pp = uvMesh.verts[vi];
		cv::Vec2f imgPos(pp[0], uv_height[0] - pp[1]);
		getPixelInterInfor(imgPos, uv_height[0], uv_Width[0],
			p0[vi], p1[vi], p2[vi], p3[vi], efi[vi]);
	}
	saveVertPixelSample(objRoot + "skirt_uv/garment_256_vertPixel.txt", p0, p1, p2, p3, efi);

}

void GeoRemap()
{
	std::string caseName = "/";
	std::string F_prefix = "D:/models/isantesteban/vto-dataset-main/dress/meshes/";
	std::string uv_folder = "dress_uv/";
	std::string objRoot = F_prefix + caseName;

}

void getGeoUV_IndexEff_2()
{
	std::string caseName = "/";
	std::string F_prefix = "D:/models/isantesteban/vto-dataset-main/dress/meshes/";
	std::string uv_folder = "dress_uv/";
	std::string objRoot = F_prefix + caseName;

	int uv_height[1] = { 256 };
	int uv_Width[1] = { 256 };

	string uvName = objRoot + uv_folder + "garment_uv.ply";
	R_Mesh uvMesh;
	readPly(uvName, uvMesh.verts, uvMesh.faceInds);

	string geoName = objRoot + uv_folder + "garment_origeo.ply";
	R_Mesh geoMesh;
	readPly(geoName, geoMesh.verts, geoMesh.faceInds);

	std::vector<cv::Vec2i> p0(geoMesh.numV());
	std::vector<cv::Vec2i> p1(geoMesh.numV());
	std::vector<cv::Vec2i> p2(geoMesh.numV());
	std::vector<cv::Vec2i> p3(geoMesh.numV());

	std::vector<cv::Vec4f> efi(geoMesh.numV());
	std::vector<cv::Vec3f> geoUV(geoMesh.numV());

	std::vector<int> geoF(geoMesh.numV(), 0);

	for (int fi = 0; fi < geoMesh.numF(); fi++)
	{
		cv::Vec3i geoFace = geoMesh.faceInds[fi];
		cv::Vec3i uvFace = uvMesh.faceInds[fi];
		for (int di = 0; di < 3; di++)
		{
			int geoID = geoFace[di];

			if (geoF[geoID] > 0)
				continue;

			cv::Vec3f uvPos = uvMesh.verts[uvFace[di]];
			geoUV[geoID] = uvPos;

			cv::Vec2f imgPos = cv::Vec2f(uvPos[0] * uv_Width[0], uv_height[0] - uvPos[1] * uv_height[0]);
			
			getPixelInterInfor(imgPos, uv_height[0], uv_Width[0],
				p0[geoID], p1[geoID], p2[geoID], p3[geoID], efi[geoID]);

			geoF[geoID] = 1;

		} // end for di
	} // end for fi

	saveVertPixelSample(objRoot + "dress_uv/garment_256_vertPixel.txt", p0, p1, p2, p3, efi);

	std::vector<cv::Vec3f> zeronormals(geoMesh.numV(), cv::Vec3f(0., 0., 0.));
	saveObjFile(F_prefix + "dress_uv/garment_geouv.obj", geoUV, zeronormals);
}

void genInterUV()
{
	std::string caseName = "/";
	std::string F_prefix = "D:/models/DS/Data_walk/mixamo_body/sequence/";
	std::string uv_folder = "skirt_uv/";
	std::string objRoot = F_prefix + caseName;

	int uv_height[1] = {128};
	int uv_Width[1] = {128};

	string uvName = objRoot + uv_folder + "garment_uv.ply";
	R_Mesh uvMesh;
	readPly(uvName, uvMesh.verts, uvMesh.faceInds);
	uvMesh.scaleMesh(uv_height[0], uv_Width[0], 1.);
	RayIntersection myTracer;
	myTracer.addObj(&uvMesh);

	string geoName = objRoot + uv_folder + "garment_geo.ply";
	R_Mesh geoMesh;
	readPly(geoName, geoMesh.verts, geoMesh.faceInds);

	//string validMaskName = objRoot + uv_folder + "body_uv_128_valid.png";
	//cv::Mat vMask = cv::imread(validMaskName, cv::IMREAD_GRAYSCALE);

	//string meshNamePre = "10_L/PD10_";

	std::vector<std::vector<cv::Vec2i>> Pixel_Valid(1);
	std::vector<std::vector<cv::Vec3i>> Pixel_VertID(1);
	std::vector<std::vector<cv::Vec2f>> Pixel_Wei(1);

	cv::Mat rstNMap = cv::Mat::zeros(uv_height[0], uv_Width[0], CV_32FC3);

	/*int fID = 2;
	char _fbuffer[8];
	std::snprintf(_fbuffer, sizeof(_fbuffer), "%07d", fID);

	R_Mesh objMesh;
	readObjVertArray(objRoot + meshNamePre + string(_fbuffer) + ".obj", objMesh.verts, 1.);
	objMesh.faceInds = geoMesh.faceInds;*/
	std::vector<int> vertFlag(geoMesh.numV(), -1);
	std::vector<int> vertIdx;
	std::vector<cv::Vec3f> validPosition;

	for (int y = 0; y < uv_height[0]; y++)
	{
		for (int x = 0; x < uv_Width[0]; x++)
		{
			cv::Vec3f ori(x, uv_height[0] - y, -10.);
			cv::Vec3f dir(0., 0., 1.);
			RTCHit h = myTracer.rayIntersection(ori, dir);
			int fID = h.primID;
			if (fID < 0)
				continue;
			else
			{
				//if (vMask.at<uchar>(y, x) > 200)
				{
					cv::Vec3i face = geoMesh.faceInds[fID];
					/*cv::Vec3f p0 = geoMesh.verts[face[0]];
					cv::Vec3f p1 = geoMesh.verts[face[1]];
					cv::Vec3f p2 = geoMesh.verts[face[2]];
					rstNMap.at<cv::Vec3f>(y, x) = (1. - h.u - h.v) * p0 + h.u * p1 + h.v * p2;*/
					for (int dd = 0; dd < 3; dd++)
						if (vertFlag[face[dd]] < 0)
						{
							vertIdx.push_back(face[dd]);
							vertFlag[face[dd]] = 1;
							validPosition.push_back(geoMesh.verts[face[dd]]);
						}
					cv::Vec2f wei(h.u, h.v);
					Pixel_Valid[0].push_back(cv::Vec2i(x, y));
					Pixel_Wei[0].push_back(wei);
					Pixel_VertID[0].push_back(face);
					rstNMap.at<cv::Vec3f>(y, x) = cv::Vec3f(1., 1., 1.);
				}
			}
		} // end for x
	} // end for y
	cv::imwrite(objRoot + uv_folder + "garment_uv_128.png", rstNMap * 255.);
	savePixelSampleMap(objRoot + uv_folder + "garment_uvMap_128.txt", 1, uv_height, uv_Width, Pixel_Wei, Pixel_VertID, Pixel_Valid);
	printf("%d \n", int(vertIdx.size()));
	saveIndexFile(objRoot + uv_folder + "garment_validIndices_128.txt", vertIdx);
}

void resaveMeshInfo()
{
	std::string caseName = "/rumba_swing/normal_body/";
	std::string F_prefix = "D:/models/DS/Data_walk/";
	//std::string objRoot = F_prefix + caseName;

	string geoName = F_prefix + "body_uv/body_geo.ply";
	R_Mesh geoMesh;
	readPly(geoName, geoMesh.verts, geoMesh.faceInds);

	geoMesh.vnormals = geoMesh.calcVertNorm();

	std::vector<cv::Vec3i> colors(geoMesh.numV());
	for (int v = 0; v < geoMesh.numV(); v++)
		colors[v] = (geoMesh.vnormals[v] + cv::Vec3f(1., 1., 1.)) * 0.5 * 255;
	savePlyFile_withNorm(F_prefix + "/body_uv/body_norm.ply", geoMesh.verts, geoMesh.vnormals, colors, geoMesh.faceInds);
	saveObjFile(F_prefix + "/body_uv/body_norm.obj", geoMesh.verts, geoMesh.vnormals);


	string meshNamePre = F_prefix + caseName + "body/b_";

	int Frame0 = 1;
	int Frame1 = 230;
	for (int fID = Frame0; fID < Frame1+1; fID++)
	{
		char _fbuffer[8];
		std::snprintf(_fbuffer, sizeof(_fbuffer), "%06d", fID);

		R_Mesh objMesh;
		readObjVertArray(meshNamePre + string(_fbuffer) + ".obj", objMesh.verts, 1.);
		objMesh.faceInds = geoMesh.faceInds;
		objMesh.vnormals = objMesh.calcVertNorm();
		saveObjFile(F_prefix + caseName + "body_n/" + string(_fbuffer) + ".obj", objMesh.verts, objMesh.vnormals);

		/*std::vector<cv::Vec3i> colors(objMesh.numV());
		for (int v = 0; v < objMesh.numV(); v++)
			colors[v] = (objMesh.vnormals[v] + cv::Vec3f(1., 1., 1.)) * 0.5 * 255;
		savePlyFile_withNorm(F_prefix + caseName + "body_n/" + string(_fbuffer) + ".ply",
			objMesh.verts, objMesh.vnormals, colors, objMesh.faceInds);
		exit(1);*/

	} // end for i

}



void canonicalLocalSpace()
{
	string fRoot = "D:/models/DS/Data_walk/mixamo_body/sequence/";
	string uvRoot = "/skirt_uv/";
	string geoName = fRoot + uvRoot + "garment_geo.ply";
	R_Mesh cancGeoModel;
	readPly(geoName, cancGeoModel.verts, cancGeoModel.faceInds);
	cancGeoModel.vnormals = cancGeoModel.calcVertNorm();
	//cancGeoModel.vnormals = read3dVectors(fRoot + "skirt_uv/Base_norm.txt");
	cancGeoModel.normalizeVertNorm();

	string uvName = fRoot + uvRoot + "garment_uv.ply";
	R_Mesh cancUVModel;
	readPly(uvName, cancUVModel.verts, cancUVModel.faceInds);

	vector<cv::Vec3f> vTangents(cancGeoModel.numV(), cv::Vec3f(0., 0., 0.));
	vector<cv::Vec3f> vBasements(cancGeoModel.numV(), cv::Vec3f(0., 0., 0.));
	vector<int> vFlags(cancGeoModel.numV(), 0);

	for (int f = 0; f < cancGeoModel.numF(); f++)
	{
		cv::Vec3i geoFace = cancGeoModel.faceInds[f];
		cv::Vec3i uvFace = cancUVModel.faceInds[f];
		for (int di = 0; di < 3; di++)
		{
			if (vFlags[geoFace[di]] > 0)
				continue;
			cv::Vec3f p = cancGeoModel.verts[geoFace[di]];
			cv::Vec3f N = cancGeoModel.vnormals[geoFace[di]];
			cv::Vec3f uv = cancUVModel.verts[uvFace[di]];

			cv::Vec3f p1 = cancGeoModel.verts[geoFace[(di + 1) % 3]];
			cv::Vec3f p2 = cancGeoModel.verts[geoFace[(di + 2) % 3]];

			cv::Vec3f uv1 = cancUVModel.verts[uvFace[(di + 1) % 3]];
			cv::Vec3f uv2 = cancUVModel.verts[uvFace[(di + 2) % 3]];

			cv::Vec3f T(0., 0., 0.), B(0., 0., 0.);
			calcTSpace(p, p1, p2,
				cv::Vec2f(uv[0], uv[1]),
				cv::Vec2f(uv1[0], uv1[1]),
				cv::Vec2f(uv2[0], uv2[1]), T);

			T = normalize(T - T.dot(N) * N);
			B = N.cross(T);

			vTangents[geoFace[di]] = T;
			vBasements[geoFace[di]] = B;
			vFlags[geoFace[di]] = 1;

		} // end for di
	} // end for f

	save3dVectors(fRoot + uvRoot + "garment_canc_normals.txt", cancGeoModel.vnormals);
	save3dVectors(fRoot + uvRoot + "garment_canc_tangents.txt", vTangents);
	save3dVectors(fRoot + uvRoot + "garment_canc_basements.txt", vBasements);

//-----------------------------------------------------------------

	int uSize = 1024;
	cancUVModel.scaleMesh(uSize, uSize, 0.);
	RayIntersection myTracer;
	myTracer.addObj(&cancUVModel);

	cv::Mat NormalMap = cv::Mat::zeros(uSize, uSize, CV_32FC3);
	cv::Mat TangentMap = cv::Mat::zeros(uSize, uSize, CV_32FC3);
	cv::Mat BasementMap = cv::Mat::zeros(uSize, uSize, CV_32FC3);
	cv::Mat maskMap = cv::Mat::zeros(uSize, uSize, CV_32FC1);

	for (int y = 0; y < uSize; y++)
	{
		for (int x = 0; x < uSize; x++)
		{
			cv::Vec3f ori(x, uSize - y, -10.);
			cv::Vec3f dir(0., 0., 1.);
			RTCHit h = myTracer.rayIntersection(ori, dir);
			int fID = h.primID;
			if (fID < 0)
				continue;
			else
			{
				cv::Vec3i face = cancGeoModel.faceInds[fID];
				cv::Vec3f n0 = cancGeoModel.vnormals[face[0]];
				cv::Vec3f n1 = cancGeoModel.vnormals[face[1]];
				cv::Vec3f n2 = cancGeoModel.vnormals[face[2]];

				cv::Vec3f t0 = vTangents[face[0]];
				cv::Vec3f t1 = vTangents[face[1]];
				cv::Vec3f t2 = vTangents[face[2]];

				cv::Vec3f b0 = vBasements[face[0]];
				cv::Vec3f b1 = vBasements[face[1]];
				cv::Vec3f b2 = vBasements[face[2]];

				cv::Vec3f nn = (1. - h.u - h.v) * n0 + h.u * n1 + h.v * n2;
				cv::Vec3f tt = (1. - h.u - h.v) * t0 + h.u * t1 + h.v * t2;
				cv::Vec3f bb = (1. - h.u - h.v) * b0 + h.u * b1 + h.v * b2;

				maskMap.at<float>(y, x) = 255;
				NormalMap.at<cv::Vec3f>(y, x) = (nn + cv::Vec3f(1., 1., 1.)) * 0.5;
				TangentMap.at<cv::Vec3f>(y, x) = (tt + cv::Vec3f(1., 1., 1.)) * 0.5;
				BasementMap.at<cv::Vec3f>(y, x) = (bb  + cv::Vec3f(1., 1., 1.)) * 0.5;

			}
		} // end for x
	} // end for y

	cv::imwrite(fRoot + uvRoot + "g_normal.png", NormalMap * 255.);
	cv::imwrite(fRoot + uvRoot + "g_tangent.png", TangentMap * 255.);
	cv::imwrite(fRoot + uvRoot + "g_base.png", BasementMap * 255.);
	cv::imwrite(fRoot + uvRoot + "g_mask.png", maskMap);
}

void getSampleInfo()
{
	std::string caseName = "/walk_75/";
	std::string F_prefix = "C:/newProj/SMPL/SMPL_mixamo/walk/";
	std::string objRoot = F_prefix + caseName;

	string geoName = objRoot + "uv/body_geo.ply";
	R_Mesh geoMesh;
	readPly(geoName, geoMesh.verts, geoMesh.faceInds);
	RayIntersection myTracer;
	myTracer.addObj(&geoMesh);

	string spntName = objRoot + "uv/body_t_1000.obj";
	vector<cv::Vec3f> samplePnts, sampleNorms;
	readObjVNArray(spntName, samplePnts, sampleNorms);
	assert(samplePnts.size() == sampleNorms.size());

	int numVs = samplePnts.size();
	vector<cv::Vec3i> sampleF(numVs);
	vector<cv::Vec2f> sampleUV(numVs);

	for (int vi = 0; vi < numVs; vi++)
	{
		cv::Vec3f p = samplePnts[vi];
		cv::Vec3f n = sampleNorms[vi];
		p = p + 0.01 * n;
		cv::Vec3f dir = -1. * n;
		RTCHit h = myTracer.rayIntersection(p, dir);
		int fID = h.primID;
		if (fID < 0)
		{
			printf("w..");
			continue;
		}
		else
		{
			cv::Vec3i face = geoMesh.faceInds[fID];
			cv::Vec2f uv = cv::Vec2f(h.u, h.v);
			sampleF[vi] = face;
			sampleUV[vi] = uv;
		}

	} // end for vi
	saveSampleInfo(objRoot + "uv/body_t_1000_sample.txt", sampleF, sampleUV);
}

void uvRegularSampleColorAssign_moreColor()
{
	std::string caseName = "/";
	std::string F_prefix = "C:/newProj/SMPL/SMPL_mixamo/walk/walk_75/";
	std::string objRoot = F_prefix + caseName;

	int uv_height[1] = { 256 };
	int uv_Width[1] = { 256 };

	string uvPref = "uv/body_uv/";

	string uvName = objRoot + uvPref + "/body_uv.ply";
	R_Mesh uvMesh;
	readPly(uvName, uvMesh.verts, uvMesh.faceInds);
	uvMesh.scaleMesh(uv_height[0], uv_Width[0], 1.);
	RayIntersection myTracer;
	myTracer.addObj(&uvMesh);

	string vColorName = objRoot + uvPref + "/moreColor.ply";
	std::vector<cv::Vec3f> vColors;
	readPlyColors(vColorName, vColors);

	string geoName = objRoot + uvPref + "/body_geo.ply";
	R_Mesh geoMesh;
	readPly(geoName, geoMesh.verts, geoMesh.faceInds);

	string validMaskName = objRoot + uvPref + "/uv_256_3c.png";
	cv::Mat vMask = cv::imread(validMaskName, cv::IMREAD_COLOR);
	vMask.convertTo(vMask, CV_32FC3);

	vector<cv::Vec3i> sampleF;
	vector<cv::Vec2f> sampleUV;
	vector<cv::Vec3f> sampleVerts;
	vector<cv::Vec3i> sampleColors;
	vector<int> sampleflagID;

	vector<cv::Vec2i> sampleUiVi;
	vector<cv::Vec3f> temuv;

	int xyi = 8;

	for (int yi = 4; yi < uv_height[0]; yi += xyi)
	{
		for (int xi = 4; xi < uv_Width[0]; xi += xyi)
		{
			cv::Vec3f ori(xi, uv_height[0] - yi, -10.);
			cv::Vec3f dir(0., 0., 1.);
			RTCHit h = myTracer.rayIntersection(ori, dir);
			int fID = h.primID;
			if (fID < 0)
				continue;
			else
			{
				cv::Vec3f mValue = vMask.at<cv::Vec3f>(yi, xi);
				float sV = mValue[0] + mValue[1] + mValue[2];
				if (sV > 0)
				{
					cv::Vec3i face = geoMesh.faceInds[fID];
					cv::Vec3f p0 = geoMesh.verts[face[0]];
					cv::Vec3f p1 = geoMesh.verts[face[1]];
					cv::Vec3f p2 = geoMesh.verts[face[2]];
					cv::Vec3f pp = (1. - h.u - h.v) * p0 + h.u * p1 + h.v * p2;
					cv::Vec2f uv = cv::Vec2f(h.u, h.v);
					sampleVerts.push_back(pp);
					sampleF.push_back(face);
					sampleUV.push_back(uv);
					sampleUiVi.push_back(cv::Vec2i(yi / xyi, xi / xyi));
					cv::Vec3i cc(0, 0, 0);
					int sidd = -1;
					temuv.push_back(cv::Vec3f(xi / float(uv_height[0]), (uv_height[0] - yi) / float(uv_Width[0]), 0.));
					if (mValue[0] > mValue[1] && mValue[0] > mValue[2])
						sidd = 2;  // body
					if (mValue[1] > mValue[0] && mValue[1] > mValue[2])
					{
						sidd = 1; // arm
					}
					if (mValue[2] > mValue[0] && mValue[2] > mValue[1])
						sidd = 0; // leg
					cc[sidd] = 255;

					cv::Vec3i uvFace = uvMesh.faceInds[fID];
					cv::Vec3f c0 = vColors[uvFace[0]];
					cv::Vec3f c1 = vColors[uvFace[1]];
					cv::Vec3f c2 = vColors[uvFace[2]];
					if ((c0[2] + c1[2] + c2[2]) < 2.5)
					{
						float cr = c0[0] + c1[0] + c2[0];
						float cg = c0[1] + c1[1] + c2[1];
						if (cr > cg)
							sidd = 3; // fore-arm
						else
							sidd = 4; // calf
						cc[0] = 255;
					}

					sampleflagID.push_back(sidd);
					sampleColors.push_back(cc);
				}
			}
		}
	}

	printf("%d", int(sampleVerts.size()));
	int sampleNum = int(sampleVerts.size());
	char _fbuffer[8];
	std::snprintf(_fbuffer, sizeof(_fbuffer), "%d", int(sampleVerts.size()));
	saveSampleInfo(objRoot + uvPref + "/body_t_" + string(_fbuffer) + "_sample_m.txt", sampleF, sampleUV, sampleflagID);
	//saveSampleInfo_uvi(objRoot + uvPref + "/body_uvi_" + string(_fbuffer) + "_sample.txt", sampleUiVi);
	vector<cv::Vec3f> tempNorm = vector<cv::Vec3f>(sampleNum, cv::Vec3f(0., 0., 0.));
	//saveObjFile(objRoot + uvPref + "/32_sample.obj", sampleVerts, tempNorm);
	savePlyFile(objRoot + uvPref + "/32_sample_uv_m.ply", temuv, sampleColors, std::vector<cv::Vec3i>());
	savePlyFile(objRoot + uvPref + "/32_sample_m.ply", sampleVerts, sampleColors, std::vector<cv::Vec3i>());

}

void uvRegularSampleColorAssign()
{
	std::string caseName = "/";
	std::string F_prefix = "D:/models/DS/Data_walk/mixamo_body/sequence/";
	std::string objRoot = F_prefix + caseName;

	int uv_height[1] = { 128 };
	int uv_Width[1] = { 128 };

	string uvPref = "dress_uv/body_uv/";

	string uvName = objRoot + uvPref + "/body_uv.ply";
	R_Mesh uvMesh;
	readPly(uvName, uvMesh.verts, uvMesh.faceInds);
	uvMesh.scaleMesh(uv_height[0], uv_Width[0], 1.);
	RayIntersection myTracer;
	myTracer.addObj(&uvMesh);

	string geoName = objRoot + uvPref + "/body_geo.ply";
	R_Mesh geoMesh;
	readPly(geoName, geoMesh.verts, geoMesh.faceInds);

	string validMaskName = objRoot + uvPref + "/body_uv_128_cc.png";
	cv::Mat vMask = cv::imread(validMaskName, cv::IMREAD_COLOR);
	vMask.convertTo(vMask, CV_32FC3);

	vector<cv::Vec3i> sampleF;
	vector<cv::Vec2f> sampleUV;
	vector<cv::Vec3f> sampleVerts;
	vector<cv::Vec3i> sampleColors;
	vector<int> sampleflagID;

	vector<cv::Vec2i> sampleUiVi;
	vector<cv::Vec3f> temuv;

	int xyi = 4;

	for (int yi = 2; yi < uv_height[0]; yi += xyi)
	{
		for (int xi = 2; xi < uv_Width[0]; xi += xyi)
		{
			cv::Vec3f ori(xi, uv_height[0] - yi, -10.);
			cv::Vec3f dir(0., 0., 1.);
			RTCHit h = myTracer.rayIntersection(ori, dir);
			int fID = h.primID;
			if (fID < 0)
				continue;
			else
			{
				cv::Vec3f mValue = vMask.at<cv::Vec3f>(yi, xi);
				float sV = mValue[0] + mValue[1] + mValue[2];
				if (sV > 0)
				{
					cv::Vec3i face = geoMesh.faceInds[fID];
					cv::Vec3f p0 = geoMesh.verts[face[0]];
					cv::Vec3f p1 = geoMesh.verts[face[1]];
					cv::Vec3f p2 = geoMesh.verts[face[2]];
					cv::Vec3f pp = (1. - h.u - h.v) * p0 + h.u * p1 + h.v * p2;
					cv::Vec2f uv = cv::Vec2f(h.u, h.v);
					sampleVerts.push_back(pp);
					sampleF.push_back(face);
					sampleUV.push_back(uv);
					sampleUiVi.push_back(cv::Vec2i(yi / xyi, xi / xyi));
					cv::Vec3i cc(0, 0, 0);
					int sidd = -1;
					temuv.push_back(cv::Vec3f(xi / float(uv_height[0]), (uv_height[0] - yi) / float(uv_Width[0]), 0.));
					if (mValue[0] > mValue[1] && mValue[0] > mValue[2])
						sidd = 2;
					if (mValue[1] > mValue[0] && mValue[1] > mValue[2])
						sidd = 1;
					if (mValue[2] > mValue[0] && mValue[2] > mValue[1])
						sidd = 0;
					cc[sidd] = 255;
					sampleflagID.push_back(sidd);
					sampleColors.push_back(cc);
				}
			}
		}
	}

	printf("%d", int(sampleVerts.size()));
	int sampleNum = int(sampleVerts.size());
	char _fbuffer[8];
	std::snprintf(_fbuffer, sizeof(_fbuffer), "%d", int(sampleVerts.size()));
	saveSampleInfo(objRoot + uvPref + "/body_t_" + string(_fbuffer) + "_sample.txt", sampleF, sampleUV, sampleflagID);
	//saveSampleInfo_uvi(objRoot + uvPref + "/body_uvi_" + string(_fbuffer) + "_sample.txt", sampleUiVi);
	vector<cv::Vec3f> tempNorm = vector<cv::Vec3f>(sampleNum, cv::Vec3f(0., 0., 0.));
	//saveObjFile(objRoot + uvPref + "/32_sample.obj", sampleVerts, tempNorm);
	savePlyFile(objRoot + uvPref + "/32_sample_uv.ply", temuv, sampleColors, std::vector<cv::Vec3i>());
	savePlyFile(objRoot + uvPref + "/32_sample.ply", sampleVerts, sampleColors, std::vector<cv::Vec3i>());

}

void uvRegularSample()
{
	std::string caseName = "/";
	std::string F_prefix = "D:/models/isantesteban/vto-dataset-main/dress/pred/";
	std::string objRoot = F_prefix + caseName;

	int uv_height[1] = { 128 };
	int uv_Width[1] = { 128 };

	string uvPref = "body_uv/";

	string uvName = objRoot + uvPref + "/body_uv.ply";
	R_Mesh uvMesh;
	readPly(uvName, uvMesh.verts, uvMesh.faceInds);
	uvMesh.scaleMesh(uv_height[0], uv_Width[0], 1.);
	RayIntersection myTracer;
	myTracer.addObj(&uvMesh);

	string geoName = objRoot + uvPref + "/body_geo.ply";
	R_Mesh geoMesh;
	readPly(geoName, geoMesh.verts, geoMesh.faceInds);

	string validMaskName = objRoot + uvPref + "/body_uv_128_noh.png";
	cv::Mat vMask = cv::imread(validMaskName, cv::IMREAD_GRAYSCALE);

	vector<cv::Vec3i> sampleF;
	vector<cv::Vec2f> sampleUV;
	vector<cv::Vec3f> sampleVerts;

	vector<cv::Vec2i> sampleUiVi;
	vector<cv::Vec3f> temuv;

	int xyi = 4;

	for (int yi = 0; yi < uv_height[0]; yi+= xyi)
	{
		for (int xi = 0; xi < uv_Width[0]; xi+= xyi)
		{
			cv::Vec3f ori(xi, uv_height[0] - yi, -10.);
			cv::Vec3f dir(0., 0., 1.);
			RTCHit h = myTracer.rayIntersection(ori, dir);
			int fID = h.primID;
			if (fID < 0)
				continue;
			else
			{
				if (vMask.at<uchar>(yi, xi) > 0)
				{
					cv::Vec3i face = geoMesh.faceInds[fID];
					cv::Vec3f p0 = geoMesh.verts[face[0]];
					cv::Vec3f p1 = geoMesh.verts[face[1]];
					cv::Vec3f p2 = geoMesh.verts[face[2]];
					cv::Vec3f pp = (1. - h.u - h.v) * p0 + h.u * p1 + h.v * p2;
					cv::Vec2f uv = cv::Vec2f(h.u, h.v);
					sampleVerts.push_back(pp);
					sampleF.push_back(face);
					sampleUV.push_back(uv);
					sampleUiVi.push_back(cv::Vec2i(yi / xyi, xi / xyi));
					temuv.push_back(cv::Vec3f(xi/float(uv_height[0]), (uv_height[0] - yi)/ float(uv_Width[0]), 0.));
				}
			}
		}
	}

	printf("%d", int(sampleVerts.size()));
	int sampleNum = int(sampleVerts.size());
	char _fbuffer[8];
	std::snprintf(_fbuffer, sizeof(_fbuffer), "%d", int(sampleVerts.size()));
	saveSampleInfo(objRoot + uvPref + "/body_t_"+string(_fbuffer) +"_sample.txt", sampleF, sampleUV);
	saveSampleInfo_uvi(objRoot + uvPref + "/body_uvi_" + string(_fbuffer) + "_sample.txt", sampleUiVi);
	vector<cv::Vec3f> tempNorm = vector<cv::Vec3f>(sampleNum, cv::Vec3f(0., 0., 0.));
	saveObjFile(objRoot + uvPref + "/32_sample.obj", sampleVerts, tempNorm);
	saveObjFile(objRoot + uvPref + "/32_sample_uv.obj", temuv, tempNorm);
}

void calcMeshLapInfo()
{
	std::string caseName = "/";
	std::string F_prefix = "C:/newProj/SMPL/SMPL_mixamo/walk/walk_75/";
	std::string objRoot = F_prefix + caseName;

	string geoName = objRoot + "uv/garment_geo.ply";
	R_Mesh geoMesh;
	readPly(geoName, geoMesh.verts, geoMesh.faceInds);

	std::vector<std::vector<int>> adjVInds = geoMesh.calcAdjVertInds();

	string meshNamePre = F_prefix + caseName + "PD10/PD10_";

	int Frame0 = 1;
	int Frame1 = 300;
	for (int fID = Frame0; fID < Frame1 + 1; fID++)
	{
		char _fbuffer[8];
		std::snprintf(_fbuffer, sizeof(_fbuffer), "%07d", fID);

		R_Mesh objMesh;
		readObjVertArray(meshNamePre + string(_fbuffer) + ".obj", objMesh.verts, 1.);
		objMesh.faceInds = geoMesh.faceInds;
		
		std::vector<cv::Vec3f> AdjVector;
		std::vector<cv::Vec3f> lapVector;
		objMesh.calcVertLaplacian(adjVInds, AdjVector, lapVector);

		saveObjFile(F_prefix + caseName + "garm_lap/" + string(_fbuffer) + ".obj", AdjVector, lapVector);

	} // end for i
}

int main()
{
	//genInterUV();
	//getUVIndexEffi();
	//getGeoUV_IndexEff_2();
	//canonicalLocalSpace();
	resaveMeshInfo();
	//getSampleInfo();
    //uvRegularSample();
	//uvRegularSampleColorAssign();
	//uvRegularSampleColorAssign_moreColor();
	//calcMeshLapInfo();

	printf("Done.\n");
	while (1);
}