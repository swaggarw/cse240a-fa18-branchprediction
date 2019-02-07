//========================================================//
//  predictor.c                                           //
//  Source file for the Branch Predictor                  //
//                                                        //
//  Implement the various branch predictors below as      //
//  described in the README                               //
//========================================================//
#include <stdio.h>
#include <stdlib.h>
#include "predictor.h"

//
// TODO:Student Information
//
const char *studentName = "SWAPNIL AGGARWAL";
const char *studentID   = "A53271788";
const char *email       = "swaggarw@ucsd.edu";

//------------------------------------//
//      Predictor Configuration       //
//------------------------------------//

// Handy Global for use in output routines
const char *bpName[4] = { "Static", "Gshare",
                          "Tournament", "Custom" };

int ghistoryBits; // Number of bits used for Global History
int lhistoryBits; // Number of bits used for Local History
int pcIndexBits;  // Number of bits used for PC index
int bpType;       // Branch Prediction Type
int verbose;

//------------------------------------//
//      Predictor Data Structures     //
//------------------------------------//

//
//TODO: Add your own Branch Predictor data structures here
//


// Variable declarations for tournament predictor

uint8_t g_pred; // Global Prediction Value
uint8_t l_pred; // Local Prediction Value
uint8_t m_pred; // Choice Prediction Value (Global/Local)
uint8_t pred; 	// Decided Prediction Value

uint8_t* mphtable; // Choice PHT Pointer

uint32_t* lhtable; // Actual Prediction Table of local predictor
uint8_t* lphtable;
uint32_t lphindex; // Address to Prediction History table of local predictor

uint32_t pcindex; // Masked PC index bits *Used in both GShare and Tournament Predictors*

uint32_t lhregindex; // Local BHT's Output Value after masking
uint32_t ghregindex; // GHR's Output Value after masking

// Variable Declarations for GShare Predictor

uint32_t gphindex; // Used to index into Global PHT
uint8_t* gphtable; // Global PHT Pointer : *Used in both GShare and Tournament Predictors*

/* Declarations for the Custom Predictor
	Implemented a Perceptron-Based Branch Predictor developed by David A. Jimemez et al (https://www.cs.utexas.edu/~lin/papers/hpca01.pdf)
	Total Memory Used: 16424 bits (< (16kb + 256b))
	
	Calculations:
	Total length of Global History Register (GHR) = 32 bits (Actual is 15 bits, but declared as 'uint32_t' type)
	Total no. of perceptrons used = psize = 128
	Total length of one perceptron = GHRsize + 1 (for bias of perceptron) = 16
	Size of one perceptron = 8 bits (declared as int8_t type)
	Size of theta (training limiter) = 8 bits (int8_t type)

	Therefore, total size of perceptron table = (psize * length of one perceptron * size of one perceptron)
											  = 128 * 16 * 8
											  = 16384 bits

	Therefore total memory used = Memory(perceptron table) + Memory(GHR) = 16384 + 32 + 8 = 16424 bits
*/

int customGBits = 15; // GHR bits for Global History Register
int customPCBits = 7; // PC bits used to index Perceptron Table

int psize = 128; // No. of perceptrons

int8_t theta; // Training Limiter

int8_t* bias_table; // Table for bias elements

int8_t** weights_table; // Pointer element for the pointer to each of the perceptron rows

int val = 0; // Value register to store the value of each calculation

int perc_addr; // Address bits to Perceptron Table


//------------------------------------//
//        Predictor Functions         //
//------------------------------------//

// Initialize the predictor
//


void init_predictor()
{
	pcindex = 0; // Used to address Global PHT by XORing with GHR in GShare and to address Local BHT in Tournament
	ghregindex = 0; // Global History Register of Gshare/Tournament Predictor
	lhregindex = 0; // Local History Register (Buffer) of Local Predictor - Used to access Local PHT
	int m = (1 << ghistoryBits); // 2^(ghistoryBits) is the size of counter table of GSHARE and Meta Predictor
	int n = (1 << pcIndexBits); // 2^(pcIndexBits) is the size of Local BHT of local predictor in Tournament
	int o = (1 << lhistoryBits); // 2^(lhistoryBits) is the size of Local PHT of local predictor in Tournament
	int i, j; // Loop Helper Variables

	switch(bpType){
		case STATIC: 
			break;
		case GSHARE:
			gphtable = malloc(m * sizeof(uint8_t)); // GShare PHT
			for(i = 0; i < m; i++)
    			gphtable[i] = (uint8_t)1; // Set all bits of GSHARE Prediction counters table as 1 (01 --> WEAKLY NOT TAKEN)
  			break;

		case TOURNAMENT:
			gphtable = malloc(m * sizeof(uint8_t)); // Global PHT
			mphtable = malloc(m * sizeof(uint8_t)); // Choice/Meta PHT 
			lhtable = malloc(n * sizeof(uint32_t)); // Local BHT //
			lphtable = malloc(o * sizeof(uint8_t)); // Local PHT //

			for(i = 0; i < m; i++)
			{
			  gphtable[i] = (uint8_t)1; // Set all bits of GSHARE PHT Counters as 1 (01 --> WEAKLY NOT TAKEN)
			  mphtable[i] = (uint8_t)1; // Set all bits of Choice/Meta PHT as 1 (01 --> WEAKLY Global)
			}

    		for(i = 0; i < n; i++)
			  lhtable[i] = (uint32_t)0; //Set all bits of Local BHT as 0 or 'NOTTAKEN'

			for(i = 0; i < o; i++)
			  lphtable[i] = (uint8_t)1; //Set all bits of Local PHT as 1 (01 --> WEAKLY NOT TAKEN)
			break;

		case CUSTOM:
    		theta = (int8_t)((1.93 * customGBits) + 7); // Mentioned in research paper implementation for optimum result for GHR > 12 bits
    		
    		ghregindex = 0; // Set all bits of GHR as not taken
    		
    		bias_table = malloc(psize*(sizeof(int8_t))); // Perceptron Bias Table 
    		
    		weights_table = malloc(psize * sizeof(int8_t*)); // Declaring 'psize' no. of rows of Perceptron Weights Table
    		
    		for(i = 0; i < psize; i++)
    			weights_table[i] = malloc(customGBits * sizeof(int8_t)); // Declaring 'customGBits' columns for each row of Perceptron Weights Table
    		
    		for(i = 0; i < psize; i++)
    		{
    			bias_table[i] = (int8_t)1; // Set all bias bits as 1
			
				for(j = 0; j < customGBits; j++)
    			{
    				weights_table[i][j] = (int8_t)0; // Set all perceptron weights as 0
    			}
    		}
		
			break;
  }
}

// make_g_pred for gshare module below
uint8_t predict_gshare(uint32_t pc)
{
	uint32_t pcmask;
	pcmask = (1 << ghistoryBits) - 1;
	 
	uint32_t ghmask;
	ghmask = (1 << ghistoryBits) - 1;
	 
	pcindex = pc & pcmask; // Select 'ghistorybits' no of PC bits
	 
	ghregindex = ghregindex & ghmask; // Masking GHR for 'ghistoryBits' no of GHR Bits

	gphindex = ghregindex ^ pcindex; // XOR PC and GHR values

	gphindex = gphindex & ghmask; // Masking to ensure no of bits are 'ghistoryBits'
	
	g_pred = (gphtable[gphindex] >> 1); // Select MSB of counter value as prediction

	return g_pred;
}

uint8_t predict_tournament(uint32_t pc)
{
	uint32_t ghmask = (1 << ghistoryBits) - 1; 		// Mask for GHR 
	uint32_t mask_pc = ((1 << pcIndexBits) - 1); 	// Mask for PC
	uint32_t mask_lh = ((1 << lhistoryBits) - 1); 	// Mask for lhistorybits

	ghregindex = ghregindex & ghmask; 				// Masking GHR for 'ghistoryBits' no of GHR Bits

	m_pred = ((mphtable[ghregindex] & 2) >> 1); 	// Select MSB of counter value as Choice/Meta prediction

 	pcindex = pc & mask_pc; 						// Select 'pcIndexBits' no of PC bits
    
    lhregindex = lhtable[pcindex]; 					// Fetch value of Local BHT at pcindex

    lhregindex = lhregindex & mask_lh; 				// Masking to ensure no of bits are 'lhistoryBits'

    l_pred = ((lphtable[lhregindex] & 2) >> 1); 	// Select MSB of Local PHT as local prediction 
    
    g_pred = ((gphtable[ghregindex] & 2) >> 1); 	// Select MSB of Global PHT as global prediction

	if(m_pred)										// Choose according to Choice/Meta Prediction (0 - GLobal, 1 - Local)
	 	pred = l_pred;
	else if(!m_pred)
	  	pred = g_pred;
	
	if(pred)
	  	return TAKEN;
	else
	  	return NOTTAKEN;
}

uint8_t predict_custom(uint32_t pc)
{	    
	uint32_t mask_pc = ((1 << customPCBits) - 1);
	uint32_t ghmask = (1 << customGBits) - 1;

	perc_addr = (pc & mask_pc); // Initial customPCBits to address the Perceptron Table

	val = 0; // Value of perceptron function

	val = bias_table[perc_addr]; // Assign bias value to val

	int i; // 'for' loop variable

	ghregindex = ghregindex & ghmask; // Get GHR value (after masking for customGBits)

	for(i = 0; i < customGBits; i++)
	{
		if(((ghregindex >> i) & 1) == 1)				 
			val = val + weights_table[perc_addr][i];	// Add weight value to val for a corresponding 1 in GHR
		else if(((ghregindex >> i) & 1) == 0)
			val = val - weights_table[perc_addr][i];	// Subtract weight value from val for a corresponding 0 in GHR
	}
	
	if(val >= 0)										// TAKEN if val >= 0, else NOTTAKEN
	  	return TAKEN;
	else
	  	return NOTTAKEN;
}

uint8_t make_prediction(uint32_t pc)
{
  switch (bpType) {
    case STATIC:
      return TAKEN;
    case GSHARE:
      return predict_gshare(pc);		// Gshare Make Prediction Function
      break;
    case TOURNAMENT:
      return predict_tournament(pc);	// Tournament Make Prediction Function
      break;
    case CUSTOM:
      return predict_custom(pc);		// Custom Make Prediction Function
      break;
    default:
      break;
  }
  // If there is not a compatable bpType then return NOTTAKEN
  return NOTTAKEN;
}

void train_gshare(uint8_t outcome)
{
	if(outcome)
	{
		if((gphtable[gphindex] == 0) || (gphtable[gphindex] == 1) || (gphtable[gphindex] == 2))	// Increment the value in Global PHT towards outcome == 1 (Saturate at 3)
		gphtable[gphindex]++;
	}
	
	else
	{
		if((gphtable[gphindex] == 2) || (gphtable[gphindex] == 3) || (gphtable[gphindex] == 1)) // Decrement the value in Global PHT towards outcome == 0 (Saturate at 0)
		gphtable[gphindex]--;
	}
	
	ghregindex = ghregindex << 1;
	ghregindex = ghregindex | outcome;	// Shift and add the outcome at LSB of GHR
}

void train_meta(uint8_t outcome)
{
  if(outcome)
  {
    if((gphtable[ghregindex] == 0) || (gphtable[ghregindex] == 1) || (gphtable[ghregindex] == 2)) // Increment the value in Global PHT towards outcome == 1 (Saturate at 3)
      gphtable[ghregindex]++;

    if((lphtable[lhregindex] == 0) || (lphtable[lhregindex] == 1) || (lphtable[lhregindex] == 2)) // Increment the value in Local PHT towards outcome == 1 (Saturate at 3)
      lphtable[lhregindex]++;
  }

  else if (!(outcome))
  {
    if((gphtable[ghregindex] == 1) || (gphtable[ghregindex] == 2) || (gphtable[ghregindex] == 3)) // Decrement the value in Global PHT towards outcome == 0 (Saturate at 0)
      gphtable[ghregindex]--;

    if((lphtable[lhregindex] == 1) || (lphtable[lhregindex] == 2) || (lphtable[lhregindex] == 3)) // Decrement the value in Local PHT towards outcome == 0 (Saturate at 0)
      lphtable[lhregindex]--;
  }

  if((l_pred != g_pred) && (l_pred == outcome))
  {
    if((mphtable[ghregindex] == 0) || (mphtable[ghregindex] == 1) || (mphtable[ghregindex] == 2)) // Increment the value in Choice/Meta PHT towards Select Local (1) (Saturate at 3)
      mphtable[ghregindex]++;
  }
  else if((l_pred != g_pred) && (g_pred == outcome))
    if((mphtable[ghregindex] == 1) || (mphtable[ghregindex] == 2) || (mphtable[ghregindex] == 3)) // Decrement the value in Choice/Meta PHT towards Select Global (0) (Saturate at 0)
      mphtable[ghregindex]--;

  uint32_t ghmask = (1 << ghistoryBits) - 1;
  uint32_t mask_lh = ((1 << lhistoryBits) - 1);

  lhtable[pcindex] =  (((lhtable[pcindex] << 1) | outcome)) & mask_lh; 	// Shift and add the outcome at LSB of Local BHT at pcindex
  ghregindex = (((ghregindex << 1)) | outcome) & ghmask;				// Shift and add the outcome at LSB of GHR
}

void train_custom(uint8_t outcome)
{
	int absval, signval;

	int i, j;
	
	absval = abs(val);						// Absolute value of 'val'

	signval = ((val >= 0) - (val < 0));		// Will return -1 for negative value, 1 for non-negative value

	if((signval != (outcome ? 1 : -1)) || (absval <= theta)) // Update weights if prediction != outcome (where outcome == 1 for taken and -1 for nottaken) or absolute 'val' is less than the training parameter 'theta' 
	{
		if((outcome == 1) && (bias_table[perc_addr] < 127))			// Update bias according to outcome, only if the value is less than 127 or greater than -127 (8 bit integer)
			bias_table[perc_addr] = bias_table[perc_addr] + 1;

		else if((outcome == 0) && (bias_table[perc_addr] > -127))
			bias_table[perc_addr] = bias_table[perc_addr] - 1;

		for(i = 0; i < customGBits; i++)							// Update perceptron weights according to outcome and corresponding GHR value, only if the value is less than 127 or greater than -127 (8 bit integer)
		{
			if((outcome == ((ghregindex >> i) & 1)) && weights_table[perc_addr][i] < 127)
				weights_table[perc_addr][i] = weights_table[perc_addr][i] + 1;
			else if(outcome != (((ghregindex & (1 << i)) >> i) & 1) && weights_table[perc_addr][i] > -127)
				weights_table[perc_addr][i] = weights_table[perc_addr][i] - 1;
		}
	}

	uint32_t ghmask = (1 << customGBits) - 1;

	ghregindex = (((ghregindex << 1) | (outcome)) & ghmask); 		// Shift and insert value of outcome into LSB of GHR
}

void train_predictor(uint32_t pc, uint8_t outcome)
{
  // Train the predictor based on the bpType
  switch (bpType) {
    case GSHARE:
    	train_gshare(outcome);	// GShare Train Predictor Function
      break;
    case TOURNAMENT:
      train_meta(outcome);		// Tournament Train Predictor Function
      break;
    case CUSTOM:
      train_custom(outcome);	// Custom Train Predictor Function
      break;
    default:;
  }
}