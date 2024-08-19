/*
	CS 6023 Assignment 3.
	Do not make any changes to the boiler plate code or the other files in the folder.
	Use cudaFree to deallocate any memory not in usage.
	Optimize as much as possible.
 */

#include "SceneNode.h"
#include <queue>
#include "Renderer.h"
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <chrono>
#include <algorithm>
__global__ void fillUp( int * gpu_h_Opacity, int *gpu_h_FinalPng, int** gpu_h_Mesh, int frameSizeX, int frameSizeY,int *  gpu_h_FrameSizeX,int * gpu_h_FrameSizeY,int * gpu_h_GlobalCoordinatesX,int * gpu_h_GlobalCoordinatesY,int V,int len){
  
  int id = blockIdx.x*blockDim.x + threadIdx.x;

  if(id < frameSizeX * frameSizeY){

        int row=id/frameSizeY;
        int col=id%frameSizeY;
        int maxi = 0;
        for(int i=0; i< V; i++){
            int x = gpu_h_GlobalCoordinatesX[i];
            int y = gpu_h_GlobalCoordinatesY[i];
            int m = x + gpu_h_FrameSizeX[i];
            int n = y + gpu_h_FrameSizeY[i];
            if(row >= x && row < m && col >=y && col < n && gpu_h_Opacity[i] > maxi){
                maxi = gpu_h_Opacity[i];
                gpu_h_FinalPng[id] = gpu_h_Mesh[i][(row-x)*gpu_h_FrameSizeY[i]+(col-y)];
            }

        }
  }
}
__global__ void dKernel(int **gpu_mp, int *gpu_translations, int numTranslations, int *gpu_h_GlobalCoordinatesX, int *gpu_h_GlobalCoordinatesY){
  
  int tid = blockIdx.x*blockDim.x + threadIdx.x;

	if(tid < numTranslations){

    int offset = tid * 3;
		int node = gpu_translations[offset];
		int *p = gpu_mp[node];
		int i=0;
        while(p[i] != -1)
		{
			int loc = p[i];
			if (gpu_translations[offset + 1] == 0 ){
                atomicSub(&gpu_h_GlobalCoordinatesX[loc], gpu_translations[offset + 2]);
            }
            else if(gpu_translations[offset + 1] == 1){
                atomicAdd(&gpu_h_GlobalCoordinatesX[loc], gpu_translations[offset + 2]);
            }
            else if(gpu_translations[offset + 1] == 2){
                atomicSub(&gpu_h_GlobalCoordinatesY[loc], gpu_translations[offset + 2]);
            }
            else{
                atomicAdd(&gpu_h_GlobalCoordinatesY[loc], gpu_translations[offset + 2]);
            }
            i++;
		}
	}
}

__global__ void BFS(int * g_fa,int * g_xa,int * g_done,int *g_hoffsetset,int * g_hcsr, int V){ //parallelising the bfs
        int tid = blockIdx.x * 1024 + threadIdx.x;
        if(tid < V){
            if(g_fa[tid] == 1 && g_xa[tid] == 0){
                g_fa[tid] = 0; // remove from depth
                g_xa[tid] = 1; // mark it visited
                int start = g_hoffsetset[tid]; 
                int end = g_hoffsetset[tid+1];
                for(int i= start;i<end;i++){
                    int loc = g_hcsr[i];
                    if(g_xa[loc] == 0){
                        g_fa[loc] = 1;
                        *g_done = 1;
                    }    
              }
            }
        }
    }


void readFile (const char *fileName, std::vector<SceneNode*> &scenes, std::vector<std::vector<int> > &edges, std::vector<std::vector<int> > &translations, int &frameSizeX, int &frameSizeY) {
	/* Function for parsing input file*/

	FILE *inputFile = NULL;
	// Read the file for input.
	if ((inputFile = fopen (fileName, "r")) == NULL) {
		printf ("Failed at opening the file %s\n", fileName) ;
		return ;
	}

	// Input the header information.
	int numMeshes ;
	fscanf (inputFile, "%d", &numMeshes) ;
	fscanf (inputFile, "%d %d", &frameSizeX, &frameSizeY) ;


	// Input all meshes and store them inside a vector.
	int meshX, meshY ;
	int globalPositionX, globalPositionY; // top left corner of the matrix.
	int opacity ;
	int* currMesh ;
	for (int i=0; i<numMeshes; i++) {
		fscanf (inputFile, "%d %d", &meshX, &meshY) ;
		fscanf (inputFile, "%d %d", &globalPositionX, &globalPositionY) ;
		fscanf (inputFile, "%d", &opacity) ;
		currMesh = (int*) malloc (sizeof (int) * meshX * meshY) ;
		for (int j=0; j<meshX; j++) {
			for (int k=0; k<meshY; k++) {
				fscanf (inputFile, "%d", &currMesh[j*meshY+k]) ;
			}
		}
		//Create a Scene out of the mesh.
		SceneNode* scene = new SceneNode (i, currMesh, meshX, meshY, globalPositionX, globalPositionY, opacity) ;
		scenes.push_back (scene) ;
	}

	// Input all relations and store them in edges.
	int relations;
	fscanf (inputFile, "%d", &relations) ;
	int u, v ;
	for (int i=0; i<relations; i++) {
		fscanf (inputFile, "%d %d", &u, &v) ;
		edges.push_back ({u,v}) ;
	}

	// Input all translations.
	int numTranslations ;
	fscanf (inputFile, "%d", &numTranslations) ;
	std::vector<int> command (3, 0) ;
	for (int i=0; i<numTranslations; i++) {
		fscanf (inputFile, "%d %d %d", &command[0], &command[1], &command[2]) ;
		translations.push_back (command) ;
	}
}


void writeFile (const char* outputFileName, int *hFinalPng, int frameSizeX, int frameSizeY) {
	/* Function for writing the final png into a file.*/
	FILE *outputFile = NULL;
	if ((outputFile = fopen (outputFileName, "w")) == NULL) {
		printf ("Failed while opening output file\n") ;
	}

	for (int i=0; i<frameSizeX; i++) {
		for (int j=0; j<frameSizeY; j++) {
			fprintf (outputFile, "%d ", hFinalPng[i*frameSizeY+j]) ;
		}
		fprintf (outputFile, "\n") ;
	}
}
int main (int argc, char **argv) {

	// Read the scenes into memory from File.
	const char *inputFileName = argv[1] ;  // shriram -- argv[1] contains input file and argv[2] contains output file
	int* hFinalPng ;

	int frameSizeX, frameSizeY ;
	std::vector<SceneNode*> scenes ;
	std::vector<std::vector<int> > edges ;
	std::vector<std::vector<int> > translations ;
	readFile (inputFileName, scenes, edges, translations, frameSizeX, frameSizeY) ; //shriram -- scenes,edges,translations vectors are filled.
	hFinalPng = (int*) malloc (sizeof (int) * frameSizeX * frameSizeY) ; // shriram -- Here just the shape is defined.

	// Make the scene graph from the matrices.
    Renderer* scene = new Renderer(scenes, edges) ; // shriram -- it is making loc list and mapping meshes to their ids.

	// Basic information.
	int V = scenes.size () ;
	int E = edges.size () ;
	int numTranslations = translations.size () ; //shriram -- just sizes

	// Convert the scene graph into a csr.
	scene->make_csr () ; // Returns the Compressed Sparse Row representation for the graph. //shriram -- it creates csr(array) and hoffsetset array
	int *hoffsetset = scene->get_h_offset () ;  // shriram -- it returns the hoffsetset that was created in prev instruction.
	int *hCsr = scene->get_h_csr () ; // shriram -- it returns the csr(array) that was created.
	int *hOpacity = scene->get_opacity () ; // hOpacity[vertexNumber] contains opacity of vertex vertexNumber. //shriram -- returns opacity of each mesh.
	int **hMesh = scene->get_mesh_csr () ; // hMesh[vertexNumber] contains the mesh attached to vertex vertexNumber. //shriram -- return the 2d array with each mesh values
	int *hGlobalCoordinatesX = scene->getGlobalCoordinatesX () ; // hGlobalCoordinatesX[vertexNumber] contains the X coordinate of the vertex vertexNumber.
	int *hGlobalCoordinatesY = scene->getGlobalCoordinatesY () ; // hGlobalCoordinatesY[vertexNumber] contains the Y coordinate of the vertex vertexNumber.
	int *hFrameSizeX = scene->getFrameSizeX () ; // hFrameSizeX[vertexNumber] contains the vertical size of the mesh attached to vertex vertexNumber.
	int *hFrameSizeY = scene->getFrameSizeY () ; // hFrameSizeY[vertexNumber] contains the horizontal size of the mesh attached to vertex vertexNumber.

	auto start = std::chrono::high_resolution_clock::now () ;

	// Code begins here.
	// Do not change anything above this comment.
	// shriram -- till now I have understood the input process.

    std::vector<std::vector<int>> mp(V);

    int * fa = (int *)malloc(sizeof(int)*V);
    memset(fa, 0, V*sizeof(int)); // depth
    
    int * xa =  (int *)malloc(sizeof(int)*V); // visited
    memset(xa,0,V*sizeof(int));

    int * g_xa = (int *) malloc(sizeof(int)*V);
    cudaMalloc(&g_xa,sizeof(int)*V);
    

    int * g_fa = (int * )malloc(sizeof(int)*(V));
    cudaMalloc(&g_fa,sizeof(int)*V);

    int * g_hcsr = (int* )malloc(sizeof(int)*E);
    cudaMalloc(&g_hcsr,sizeof(int)*E);
    cudaMemcpy(g_hcsr,hCsr,sizeof(int)*E,cudaMemcpyHostToDevice);

    int * g_hoffsetset = (int*)malloc(sizeof(int)*(V+1));
    cudaMalloc(&g_hoffsetset,sizeof(int)*(V+1));
    cudaMemcpy(g_hoffsetset,hoffsetset,sizeof(int)*(V+1),cudaMemcpyHostToDevice);

    for (int i=0;i<V;i++){
         memset(xa,0,sizeof(int)*V);
         memset(fa,0,sizeof(int)*V);
        fa[i] = 1;
        cudaMemcpy(g_fa,fa,sizeof(int)*V,cudaMemcpyHostToDevice);
        xa[i] = 0;
        cudaMemcpy(g_xa,xa,sizeof(int)*V,cudaMemcpyHostToDevice);
        int done = 0;
        int * g_done;
        int blocks = ceil(V/1024.0);
        cudaMalloc(&g_done,sizeof(int));
        do {
             done = 0;

            cudaMemcpy(g_done,&done,sizeof(int),cudaMemcpyHostToDevice);


            BFS<<< blocks,1024>>> (g_fa,g_xa,g_done,g_hoffsetset,g_hcsr,V);
            cudaDeviceSynchronize();

            cudaMemcpy(&done,g_done,sizeof(int),cudaMemcpyDeviceToHost);

         } while(done);

         cudaMemcpy(xa,g_xa,sizeof(int)*V,cudaMemcpyDeviceToHost);
         for(int k = 0 ; k<V; k++ ){
            if(xa[k] == 1){
            mp[i].push_back(k);
            }
         }
         mp[i].push_back(-1);
        } 


   int **gpu_mp;
	cudaMalloc(&gpu_mp, sizeof(int *) * V);


    for (int i = 0; i < V; i++)
	{
		 int s = mp[i].size();

		int *temp;
		cudaMalloc(&temp, sizeof(int) * s);
		cudaMemcpy(temp, mp[i].data(), sizeof(int) * s, cudaMemcpyHostToDevice);

		cudaMemcpy(gpu_mp + i, &temp, sizeof(int *), cudaMemcpyHostToDevice);
	}

   int *gpu_translations;
	cudaMalloc(&gpu_translations, sizeof(int) * numTranslations * 3);

	for (int i = 0; i < numTranslations; i++)
		cudaMemcpy(gpu_translations + i * 3, translations[i].data(), 3 * sizeof(int), cudaMemcpyHostToDevice);


   int *gpu_h_GlobalCoordinatesX;
	cudaMalloc(&gpu_h_GlobalCoordinatesX,sizeof(int) * V);
    cudaMemcpy(gpu_h_GlobalCoordinatesX,hGlobalCoordinatesX,sizeof(int)*V, cudaMemcpyHostToDevice);
 

	int *gpu_h_GlobalCoordinatesY;
	cudaMalloc(&gpu_h_GlobalCoordinatesY, sizeof(int) * V);
	cudaMemcpy(gpu_h_GlobalCoordinatesY, hGlobalCoordinatesY, sizeof(int) * V, cudaMemcpyHostToDevice);

   
    int b1 = ceil(numTranslations / 1024.0);
	dKernel<<<b1, 1024>>>(gpu_mp, gpu_translations, numTranslations, gpu_h_GlobalCoordinatesX, gpu_h_GlobalCoordinatesY);
    cudaDeviceSynchronize();

	cudaFree(gpu_mp);
	cudaFree(gpu_translations);

    int * gpu_h_Opacity;
    cudaMalloc(&gpu_h_Opacity,sizeof(int)*V);
    cudaMemcpy(gpu_h_Opacity,hOpacity,sizeof(int)*V,cudaMemcpyHostToDevice);

    int ** gpu_h_Mesh;
	cudaMalloc(&gpu_h_Mesh,V * sizeof(int*));
	for(int i=0;i<V;i++){
		int * temp;
		int len = hFrameSizeX[i]*hFrameSizeY[i];
		cudaMalloc(&temp,len*sizeof(int));
		cudaMemcpy(temp,hMesh[i],len*sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_h_Mesh+i,&temp,sizeof(int*),cudaMemcpyHostToDevice);
	}
    memset(hFinalPng, 0, frameSizeX*frameSizeY*sizeof(int));

    int * gpu_h_FinalPng;
	cudaMalloc(&gpu_h_FinalPng,frameSizeX*frameSizeY*sizeof(int));
	cudaMemcpy(gpu_h_FinalPng,hFinalPng,frameSizeX*frameSizeY*sizeof(int),cudaMemcpyHostToDevice);

    int * gpu_h_FrameSizeX;
    cudaMalloc(&gpu_h_FrameSizeX,sizeof(int)*V);
    cudaMemcpy(gpu_h_FrameSizeX,hFrameSizeX,sizeof(int)*V,cudaMemcpyHostToDevice);

    int * gpu_h_FrameSizeY;
    cudaMalloc(&gpu_h_FrameSizeY,sizeof(int)*V);
    cudaMemcpy(gpu_h_FrameSizeY,hFrameSizeY,sizeof(int)*V,cudaMemcpyHostToDevice);
    int len = frameSizeX*frameSizeY;
    int b2 = ceil(len/1024.0);

    fillUp <<<b2,1024>>> (gpu_h_Opacity, gpu_h_FinalPng, gpu_h_Mesh, frameSizeX, frameSizeY,  gpu_h_FrameSizeX, gpu_h_FrameSizeY,gpu_h_GlobalCoordinatesX,gpu_h_GlobalCoordinatesY,V,len);
    cudaDeviceSynchronize();

    cudaMemcpy(hFinalPng,gpu_h_FinalPng,sizeof(int)*frameSizeX*frameSizeY,cudaMemcpyDeviceToHost);


    cudaFree(gpu_h_FinalPng);
    cudaFree(gpu_h_FrameSizeX);
    cudaFree(gpu_h_FrameSizeY);
    cudaFree(gpu_h_Mesh);

	auto end  = std::chrono::high_resolution_clock::now () ;

	std::chrono::duration<double, std::micro> timeTaken = end-start;

	printf ("execution time : %f\n", timeTaken) ;
	// Write output matrix to file.
	const char *outputFileName = argv[2] ;
	writeFile (outputFileName, hFinalPng, frameSizeX, frameSizeY) ;
	return 0;
}
