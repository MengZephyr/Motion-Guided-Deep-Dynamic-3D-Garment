#pragma once

#include "opencv/cv.h"

#include <vector>
#include <algorithm>

struct KDTreeNode {
	static const size_t IsLeaf = ( (size_t)( -1 ) ) ^ ( ( (size_t)( -1 ) ) >> 1 );

	size_t next;
	unsigned axis;
	float split;
};

struct KDTreeLeaf {
	size_t id;
	cv::Vec3f pos;
};

class KDTree {
private:
	std::vector< KDTreeNode > nodes;
	std::vector< KDTreeLeaf > leaves;

public:
	KDTree() {
	}

	KDTree( const std::vector<cv::Vec3f> &points ) {
		this->buildTree( points );
	}

private:
	unsigned nextAxis( void ) {
		static unsigned i = 12345678;
		i = i * 0xaffa8375UL + 23333333;
		return ((i >> 27) ^ i) % 3;
	}

	size_t recurBuild(
		const size_t bp,
		const size_t ep,
		const unsigned axis
		) {

		auto comp = [&axis](const KDTreeLeaf& a, const KDTreeLeaf& b) -> bool {return a.pos[axis] < b.pos[axis];};

		KDTreeNode node;
		size_t idnow;

		size_t mid = (bp + ep) >> 1;

		node.axis = axis;

		idnow = nodes.size();
		nodes.push_back(node);
		

		if (bp + 1 != ep) {
			std::sort(leaves.begin() + bp, leaves.begin() + ep, comp);

			nodes[idnow].split = 0.5f * ( leaves[mid].pos[axis] + leaves[mid-1].pos[axis] );

			recurBuild(bp, mid, this ->nextAxis());
			nodes[idnow].next = recurBuild(mid, ep, this ->nextAxis());
		}
		else {
			nodes[idnow].next = (KDTreeNode::IsLeaf) | (bp);
		}

		return idnow;
	}

public:
	void buildTree( const std::vector<cv::Vec3f> &points ) {
		this->nodes.clear();
		this->leaves.resize(points.size());
		this->nodes.reserve(points.size() * 2);

		for ( size_t i = 0; i < points.size ( ); ++ i ) {
			this->leaves[i].id = i;
			this->leaves[i].pos = points[i];
		}

		this->recurBuild(0, leaves.size(), this->nextAxis());
		return;
	}

private:
	size_t recurSearch ( const size_t id, const cv::Vec3f& pos, float &minDist ) const {
		if (nodes[id].next & nodes[id].IsLeaf) {
			size_t res = nodes[id].next ^ nodes[id].IsLeaf;

			if (norm( leaves[res].pos - pos ) < minDist) {
				minDist = norm( leaves[res].pos - pos ) ;

				return res;
			}
			else
				return -1;
		}
		else {
			size_t nid, bid, nn, bnn;

			nid = (pos[nodes[id].axis] > nodes[id].split) ? nodes[id].next : id + 1;
			
			nn = recurSearch( nid, pos, minDist );
			
			if (minDist > abs( nodes[id].split - pos[nodes[id].axis] )) {
				bid = nodes[id].next ^ (id + 1) ^ nid;
				bnn = recurSearch( bid, pos, minDist );
			}
			else {
				bnn = -1;
			}

			return (bnn == -1) ? nn : bnn;
		}
	}

	void recurSearchFixedRadius ( const size_t id, const cv::Vec3f& pos, const float minDist, std::vector< size_t >& results ) const {
		if ( nodes[id].next & nodes[id].IsLeaf ) {
			size_t res = nodes[id].next ^ nodes[id].IsLeaf;

			if ( norm( leaves[res].pos - pos ) < minDist ) {
				results.push_back ( res );
			}
		}
		else {
			size_t nid, bid;

			nid = ( pos[nodes[id].axis] > nodes[id].split ) ? nodes[id].next : id + 1;

			recurSearchFixedRadius ( nid, pos, minDist, results);

			if ( minDist >= abs ( nodes[id].split - pos[nodes[id].axis] ) ) {
				bid = nodes[id].next ^ ( id + 1 ) ^ nid;
				recurSearchFixedRadius ( bid, pos, minDist, results);
			}
		}
		return;
	}

	void recurSearchKNN ( const size_t id, const cv::Vec3f& pos, const size_t K, std::vector< size_t >& results ) const {
		if ( nodes[id].next & nodes[id].IsLeaf ) {
			size_t res = nodes[id].next ^ nodes[id].IsLeaf;

			if ( results.size ( ) < K || norm( leaves[res].pos - pos ) < norm ( leaves[results[results.size ( ) - 1]].pos - pos ) ) {
				std::vector<size_t>::iterator it = results.begin ( );
				while ( it != results.end ( ) && norm( leaves[*it].pos - pos ) <= norm ( leaves[res].pos - pos ) )
					++ it;
				results.insert ( it, res );
				if ( results.size ( ) > K ) 
					results.pop_back ( );
			}
		}
		else {
			size_t nid, bid;

			nid = ( pos[nodes[id].axis] > nodes[id].split ) ? nodes[id].next : id + 1;

			recurSearchKNN ( nid, pos, K, results );

			if ( results.size() < K || norm( leaves[ *(results.rbegin()) ].pos - pos ) >= abs ( nodes[id].split - pos[nodes[id].axis] ) ) {
				bid = nodes[id].next ^ ( id + 1 ) ^ nid;
				recurSearchKNN ( bid, pos, K, results );
			}
		}
		return;
	}

public:
	KDTreeLeaf search( const cv::Vec3f& pos ) const {
		float minDist = 1e30f;
		size_t res = recurSearch ( 0, pos, minDist );
		return leaves[res];
	}

	std::vector <KDTreeLeaf> searchInRadius ( const cv::Vec3f& pos, const float radius ) const {
		std::vector < size_t > a;
		std::vector <KDTreeLeaf> res;

		recurSearchFixedRadius ( 0, pos, radius, a );
		
		res.reserve ( a.size ( ) );
		for ( size_t i : a )
			res.push_back ( leaves[i] );
		return res;
	}

	std::vector <KDTreeLeaf> searchKNN ( const cv::Vec3f& pos, const size_t K ) const {
		std::vector < size_t > a;
		std::vector <KDTreeLeaf> res;

		recurSearchKNN ( 0, pos, K, a );

		res.reserve ( a.size ( ) );
		for ( size_t i : a )
			res.push_back ( leaves[i] );
		return res;
	}

	KDTreeLeaf bruteForceSearch( const cv::Vec3f& pos ) const {
		float minDist = 1e30f;
		size_t minid;
		for ( size_t i = 0; i < leaves.size ( ); ++ i ) {
			if (norm(leaves[i].pos - pos) < minDist) {
				minDist = norm( leaves[i].pos - pos );
				minid = i;
			}
		}

		return leaves[minid];
	}
};

/*
float frand() {
	static unsigned i = 12345678;
	i = i * 0xaffa8375UL + 23323333;
	return (float)((( i >> 25 ) ^ (i)) & 0x7fffff ) / ( 0x800000 );
}

int main() {
	std::vector< glm::vec3 > a;
	std::vector< glm::vec3 > b;
	a.clear();
	b.clear();

	for (int i = 0; i < 20000; ++ i) {
		a.push_back( glm::vec3(frand(), frand(), frand()) );
		b.push_back( glm::vec3(frand(), frand(), frand()) );
	}

	KDTree mkdt( a );
	
	size_t t1, t2, t3;

	for (int i = 0; i < b.size(); ++ i) {
		KDTreeLeaf r1 = mkdt.search( b[i] );
		KDTreeLeaf r2 = mkdt.bruteForceSearch( b[i] );

		if (r1.id != r2.id) 
			//std::cout << std::endl;
			std::cout << r1.id << " " << r2.id << std::endl;
	}

	t1 = std::clock();
	for (int i = 0; i < b.size(); ++ i) {
		KDTreeLeaf r2 = mkdt.bruteForceSearch( b[i] );
	}

	t2 = std::clock();

	for (int i = 0; i < b.size(); ++ i) {
		KDTreeLeaf r1 = mkdt.search( b[i] );
	}
	t3 = std::clock();

	std::cout << (double)(t2-t1) / CLOCKS_PER_SEC << std::endl;
	std::cout << (double)(t3-t2) / CLOCKS_PER_SEC << std::endl;
	
	return 0;
}*/