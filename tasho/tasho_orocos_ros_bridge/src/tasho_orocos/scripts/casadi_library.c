/*
 *    This file is part of CasADi.
 *
 *    CasADi -- A symbolic framework for dynamic optimization.
 *    Copyright (C) 2010-2014 Joel Andersson, Joris Gillis, Moritz Diehl,
 *                            K.U. Leuven. All rights reserved.
 *    Copyright (C) 2011-2014 Greg Horn
 *
 *    CasADi is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU Lesser General Public
 *    License as published by the Free Software Foundation; either
 *    version 3 of the License, or (at your option) any later version.
 *
 *    CasADi is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public
 *    License along with CasADi; if not, write to the Free Software
 *    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */
#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"

#include <stdio.h>

#include <casadi/casadi_c.h>

//while trying to compile with lua
//g++ mp_cooperative_c.c -I/usr/local/include/casadi/ -fPIC -std=c++11 -L/usr/local/lib -L/home/ajay/home/ajay/Downloads/lua-5.1.5/src -I/home/ajay/home/ajay/Downloads/lua-5.1.5/src -llua -lcasadi -lstdc++

//compilation that works
//g++ mp_cooperative_c.c -I/usr/local/include/casadi/ -fPIC -std=c++11 -L/usr/local/lib -lcasadi -lstdc++

//converted to simply a c file
////gcc mp_cooperative_c.c -I/usr/local/include/casadi/ -fPIC -L/usr/local/lib -lcasadi

//now trying to compile this c file with some lua files
//gcc mp_cooperative_c.c -I/usr/local/include/casadi/ -fPIC -L/usr/local/lib -L/home/ajay/home/ajay/Downloads/lua-5.1.5/src -I/home/ajay/home/ajay/Downloads/lua-5.1.5/src -llua -lcasadi 

//compilation together that worked
//gcc -Wall -pedantic -fPIC -shared -llua  mp_cooperative_c.c -I/usr/local/include/casadi/ -fPIC -L/usr/local/lib -L/home/ajay/home/ajay/Downloads/lua-5.1.5/src -I/home/ajay/home/ajay/Downloads/lua-5.1.5/src  -lcasadi -o motionplanner.so


// Usage from C
int usage_c(double *sol){
  printf("---\n");
  printf("Standalone usage from C/C++:\n");
  printf("\n");

  // Sanity-check on integer type
  if (casadi_c_int_width()!=sizeof(casadi_int)) {
    printf("Mismatch in integer size\n");
    return -1;
  }
  if (casadi_c_real_width()!=sizeof(double)) {
    printf("Mismatch in double size\n");
    return -1;
  }

  // Push Function(s) to a stack
  int ret = casadi_c_push_file("/home/ajay/Desktop/first_task_wilm/yumi_laser_tracing/motion_planner.casadi");
  if (ret) {
    printf("Failed to load file 'f.casadi'.\n");
    return -1;
  }
  
  printf("Loaded number of functions: %d\n", casadi_c_n_loaded());

  // Identify a Function by name
  int id = casadi_c_id("f");


  casadi_int n_in = casadi_c_n_in_id(id);
  casadi_int n_out = casadi_c_n_out_id(id);

  casadi_int sz_arg=n_in, sz_res=n_out, sz_iw=0, sz_w=0;

  casadi_c_work_id(id, &sz_arg, &sz_res, &sz_iw, &sz_w);
  printf("Work vector sizes:\n");
  printf("sz_arg = %lld, sz_res = %lld, sz_iw = %lld, sz_w = %lld\n\n",
         sz_arg, sz_res, sz_iw, sz_w);

  // /* Print the sparsities of the inputs and outputs */
  // casadi_int i;
  // for(i=0; i<n_in + n_out; ++i){
  //   // Retrieve the sparsity pattern - CasADi uses column compressed storage (CCS)
  //   const casadi_int *sp_i;
  //   if (i<n_in) {
  //     printf("Input %lld\n", i);
  //     sp_i = casadi_c_sparsity_in_id(id, i);
  //   } else {
  //     printf("Output %lld\n", i-n_in);
  //     sp_i = casadi_c_sparsity_out_id(id, i-n_in);
  //   }
  //   if (sp_i==0) return 1;
  //   casadi_int nrow = *sp_i++; /* Number of rows */
  //   casadi_int ncol = *sp_i++; /* Number of columns */
  //   const casadi_int *colind = sp_i; /* Column offsets */
  //   const casadi_int *row = sp_i + ncol+1;  Row nonzero 
  //   casadi_int nnz = sp_i[ncol]; /* Number of nonzeros */

  //   /* Print the pattern */
  //   printf("  Dimension: %lld-by-%lld (%lld nonzeros)\n", nrow, ncol, nnz);
  //   printf("  Nonzeros: {");
  //   casadi_int rr,cc,el;
  //   for(cc=0; cc<ncol; ++cc){                    /* loop over columns */
  //     for(el=colind[cc]; el<colind[cc+1]; ++el){ /* loop over the nonzeros entries of the column */
  //       if(el!=0) printf(", ");                  /* Separate the entries */
  //       rr = row[el];                            /* Get the row */
  //       printf("{%lld,%lld}",rr,cc);                 /* Print the nonzero */
  //     }
  //   }
  //   printf("}\n\n");
  // }

  /* Allocate input/output buffers and work vectors*/
  const double *arg[sz_arg];
  double *res[sz_res];
  casadi_int iw[sz_iw];
  double w[sz_w];

  /* Function input and output */
  double x_val[1063];
  for (int i=0;i<1063;++i) {
  	x_val[i] = 0;
  }
  printf("input 1 : %g\n",x_val[1062]);
  double y_val[0];
  double res0[1062];
  //double res1[4];

  // Allocate memory (thread-safe)
  casadi_c_incref_id(id);

  /* Evaluate the function */
  arg[0] = x_val;
  arg[1] = y_val;
  res[0] = res0;


  // Checkout thread-local memory (not thread-safe)
  int mem = casadi_c_checkout_id(id);

  // Evaluation is thread-safe
  if (casadi_c_eval_id(id, arg, res, iw, w, mem)) return 1;

  // Release thread-local (not thread-safe)
  casadi_c_release_id(id, mem);

  /* Print result of evaluation */
  printf("result (0): %g\n",res0[0]);
  for (int i=0;i<1062;++i) {
  	sol[i] = res0[i];
  }
  //printf("result (1): [%g,%g;%g,%g]\n",res1[0],res1[1],res1[2],res1[3]);

  /* Free memory (thread-safe) */
  casadi_c_decref_id(id);

  // Clear the last loaded Function(s) from the stack
  casadi_c_pop();

  return 0;
}

static int hello_world(lua_State* L){
  printf("Hello world\n");
  double sol[1062];
  usage_c(sol);
  printf("sol[0] = %f and sol[1] = %f and sol[2] = %f \n", sol[0], sol[1], sol[2]);
  int i, n;
  
  // 1st argument must be a table (t)
  luaL_checktype(L,1, LUA_TTABLE);

  n = luaL_getn(L, 2);  // get size of table

  double r[n];
  double s[n];
double res[n];

  for (i=1; i<=n; i++)
{		
	lua_rawgeti(L, 1, i);  // push t
	int Top = lua_gettop(L);
	r[i-1] = lua_tonumber(L, Top);
	lua_pop(L,1);	
}
// n = luaL_getn(L, 2);  // get size of table
// for (i=1; i<=n; i++)
// {		
// 	lua_rawgeti(L, 2, i);  // push t
// 	int Top = lua_gettop(L);
// 	s[i-1] = lua_tonumber(L, Top);
// 	lua_pop(L,1);	
// }


// for(i = 0; i < n; i ++){
// 	res[i] = r[i] + s[i];
// }

//printf("val r[3] = %f and s[3] = %f and sum res[3] = %f \n", r[2], s[2], res[2]);
    printf("Do you dump here?\n");
  lua_newtable(L);
  for(i = 1; i<=1062; i++){
  	
  	lua_pushnumber(L, sol[i-1]);
  	lua_rawseti(L, -2, i);
  	//int Top = lua_gettop(L);
  }

  //lua_pushnumber(L, res[0]);
  return 1;
}

static int ocp_fun(lua_State* L){
  printf("Loading the OCP function\n");

  int ret = casadi_c_push_file("/home/ajay/Desktop/casadi_libaries_from_tasho/yumi_laser_contouring/ocp_fun.casadi");
  if (ret) {
    printf("Failed to load file 'f.casadi'.\n");
    return -1;
  }
  
  printf("Loaded number of functions: %d\n", casadi_c_n_loaded());
  
  // Identify a Function by name
  int id = casadi_c_id("ocp_fun");

  //creating input vectors and buffers for calling casadi function
  casadi_int n_in = casadi_c_n_in_id(id);
  casadi_int n_out = casadi_c_n_out_id(id);
  casadi_int sz_arg=n_in, sz_res=n_out, sz_iw=0, sz_w=0;

  casadi_c_work_id(id, &sz_arg, &sz_res, &sz_iw, &sz_w);
  printf("Work vector sizes:\n");
  printf("sz_arg = %lld, sz_res = %lld, sz_iw = %lld, sz_w = %lld\n\n",
         sz_arg, sz_res, sz_iw, sz_w);
  const double *arg[sz_arg];
  double *res[sz_res];
  casadi_int iw[sz_iw];
  double w[sz_w];
  
  /* Function input and output */
 
  int i = 0;
  int j = 0;
  int no_args = 11;
  int n[11] = {198, 198, 11, 11, 10, 180, 18, 18, 1, 1, 1108}; //18*20*2 + 18*20*19 - joint pose, velocity and acceleration vectors
  int n_params = 102;
  double temp[1100];
  double* x_0[no_args]; //initial guess for the solver
    //Allocate memory for all the input arguments to the casadi functions
  double temp0[198];
  double temp1[198];
  double temp2[11];
  double temp3[11];
  double temp4[10];
  double temp5[180];
  double temp6[18];
  double temp7[18];
  double temp8[1];
  double temp9[1];
  double temp10[1108];

  x_0[0] = temp0;
  x_0[1] = temp1;
  x_0[2] = temp2;
  x_0[3] = temp3;
  x_0[4] = temp4;
  x_0[5] = temp5;
  x_0[6] = temp6;
  x_0[7] = temp7;
  x_0[8] = temp8;
  x_0[9] = temp9;
  x_0[10] = temp10;

  /* Evaluate the function */
  for (i = 0; i<11; i++){
    arg[i] = x_0[i];
  }

  // double rtemp0[198];
  // double rtemp1[198];
  // double rtemp2[11];
  // double rtemp3[11];
  // double rtemp4[10];
  // double rtemp5[180];
  // double rtemp6[18];
  // double rtemp7[18];
  // double rtemp8[1];
  // double rtemp9[1];
  // double rtemp10[1108];

  // res[0] = rtemp0;
  // res[1] = rtemp1;
  // res[2] = rtemp2;
  // res[3] = rtemp3;
  // res[4] = rtemp4;
  // res[5] = rtemp5;
  // res[6] = rtemp6;
  // res[7] = rtemp7;
  // res[8] = rtemp8;
  // res[9] = rtemp9;
  // res[10] = rtemp10;

  // for(i = 0; i<11; i++){
  //   double x[n[i]];
  //   double res_element[n[i]];
  //   // x_0[i] = x;
  //   results[i] = res_element;
  // }
  // double params[102]; //input for the parameters of the ocp
  double result[1754]; //output of the solver
  res[0] = result;
   

  //read the initial guess from lua 

  
 



  for (j = 1; j<= no_args; j++){

     luaL_checktype(L,j, LUA_TTABLE); // 1st argument must be a table (t)
    
    for (i=1; i<=n[j-1]; i++){		
	    lua_rawgeti(L, j, i);  // push t
	    int Top = lua_gettop(L);
	    x_0[j-1][i-1] = lua_tonumber(L, Top);
	    lua_pop(L,1);	
    }
  }


  // printf("val result[1] = %f and res[2] = %f and res[3] = %f \n", x_0[6][0], result[1], result[2]);
 //  luaL_checktype(L,2, LUA_TTABLE); // 1st argument must be a table (t)
 //  for (i=1; i<=n_params; i++){		
	// lua_rawgeti(L, 2, i);  // push t
	// int Top = lua_gettop(L);
	// params[i-1] = lua_tonumber(L, Top);
	// lua_pop(L,1);	
 //  }



  // Allocate memory (thread-safe)
  casadi_c_incref_id(id);
  int mem = casadi_c_checkout_id(id);

  
  // arg[0] = x_0;
  // arg[1] = params;
  // for (i = 0; i<11; i++){
  //   // res[i] = results[i];
  //   res[i] = x_0[i];
  // }
  // res[0] = result;

  // Evaluation is thread-safe
  if (casadi_c_eval_id(id, arg, res, iw, w, mem)) return 1;
  printf("This ran\n");
  // Release thread-local (not thread-safe)
  // casadi_c_release_id(id, mem);



  printf("val result[1] = %f and res[2] = %f and res[3] = %f \n", res[0][396], res[0][397], res[0][398]);
  lua_newtable(L); //creating a lua table to pass the solution to lua
  // int k = 1;
  // for(i = 0; i<no_args; i++){

  //   for (j = 0; j<n[i];j++){
  	
  //  	lua_pushnumber(L, res[i][j]);
  //  	lua_rawseti(L, -2, k);
  //   k = k +1;
  //   }
  // }

  for(i = 1; i<=1754; i++){
    
    lua_pushnumber(L, result[i-1]);
    lua_rawseti(L, -2, i);
   }


   /* Free memory (thread-safe) */
  casadi_c_decref_id(id);

  // Clear the last loaded Function(s) from the stack
  casadi_c_pop();

  return 1;
}

static int mp_rightarm(lua_State* L){
  printf("Starting the motion planner for the right arm\n");

  int ret = casadi_c_push_file("/home/ajay/Desktop/first_task_wilm/yumi_laser_tracing/mp_right_arm.casadi");
  if (ret) {
    printf("Failed to load file 'f.casadi'.\n");
    return -1;
  }
  
  printf("Loaded number of functions: %d\n", casadi_c_n_loaded());

  // Identify a Function by name
  int id = casadi_c_id("f");

  //creating input vectors and buffers for calling casadi function
  casadi_int n_in = casadi_c_n_in_id(id);
  casadi_int n_out = casadi_c_n_out_id(id);
  casadi_int sz_arg=n_in, sz_res=n_out, sz_iw=0, sz_w=0;

  casadi_c_work_id(id, &sz_arg, &sz_res, &sz_iw, &sz_w);
  printf("Work vector sizes:\n");
  printf("sz_arg = %lld, sz_res = %lld, sz_iw = %lld, sz_w = %lld\n\n",
         sz_arg, sz_res, sz_iw, sz_w);
  const double *arg[sz_arg];
  double *res[sz_res];
  casadi_int iw[sz_iw];
  double w[sz_w];

  /* Function input and output */
  int n = 1062; //18*20*2 + 18*20*19 - joint pose, velocity and acceleration vectors
  int n_params = 102;
  double x_0[1062]; //initial guess for the solver
  double params[102]; //input for the parameters of the ocp
  double result[1062]; //output of the solver
  int i = 0;

  //read the initial guess from lua 
  
  luaL_checktype(L,1, LUA_TTABLE); // 1st argument must be a table (t)

  for (i=1; i<=n; i++){		
	lua_rawgeti(L, 1, i);  // push t
	int Top = lua_gettop(L);
	x_0[i-1] = lua_tonumber(L, Top);
	lua_pop(L,1);	
  }

  luaL_checktype(L,2, LUA_TTABLE); // 1st argument must be a table (t)
  for (i=1; i<=n_params; i++){		
	lua_rawgeti(L, 2, i);  // push t
	int Top = lua_gettop(L);
	params[i-1] = lua_tonumber(L, Top);
	lua_pop(L,1);	
  }



  // Allocate memory (thread-safe)
  casadi_c_incref_id(id);
  int mem = casadi_c_checkout_id(id);

  /* Evaluate the function */
  arg[0] = x_0;
  arg[1] = params;
  res[0] = result;

  // Evaluation is thread-safe
  if (casadi_c_eval_id(id, arg, res, iw, w, mem)) return 1;

  // Release thread-local (not thread-safe)
  casadi_c_release_id(id, mem);



  printf("val result[1] = %f and res[2] = %f and res[3] = %f \n", result[0], result[1], result[2]);
  lua_newtable(L); //creating a lua table to pass the solution to lua
  for(i = 1; i<=n; i++){
  	
   	lua_pushnumber(L, result[i-1]);
   	lua_rawseti(L, -2, i);
   }

   /* Free memory (thread-safe) */
  casadi_c_decref_id(id);

  // Clear the last loaded Function(s) from the stack
  casadi_c_pop();

  return 1;
}

static int mp_dualarm(lua_State* L){
  printf("Starting the motion planner for the right arm\n");

  int ret = casadi_c_push_file("/home/ajay/Desktop/first_task_wilm/yumi_laser_tracing/mp_dual_arm_cooperative.casadi");
  if (ret) {
    printf("Failed to load file 'f.casadi'.\n");
    return -1;
  }
  
  printf("Loaded number of functions: %d\n", casadi_c_n_loaded());

  // Identify a Function by name
  int id = casadi_c_id("f");

  //creating input vectors and buffers for calling casadi function
  casadi_int n_in = casadi_c_n_in_id(id);
  casadi_int n_out = casadi_c_n_out_id(id);
  casadi_int sz_arg=n_in, sz_res=n_out, sz_iw=0, sz_w=0;

  casadi_c_work_id(id, &sz_arg, &sz_res, &sz_iw, &sz_w);
  printf("Work vector sizes:\n");
  printf("sz_arg = %lld, sz_res = %lld, sz_iw = %lld, sz_w = %lld\n\n",
         sz_arg, sz_res, sz_iw, sz_w);
  const double *arg[sz_arg];
  double *res[sz_res];
  casadi_int iw[sz_iw];
  double w[sz_w];

  /* Function input and output */
  int n = 1062; //18*20*2 + 18*20*19 - joint pose, velocity and acceleration vectors
  int n_params = 52;
  double x_0[1062]; //initial guess for the solver
  double params[52]; //input for the parameters of the ocp
  double result[1062]; //output of the solver
  int i = 0;

  //read the initial guess from lua 
  
  luaL_checktype(L,1, LUA_TTABLE); // 1st argument must be a table (t)

  for (i=1; i<=n; i++){		
	lua_rawgeti(L, 1, i);  // push t
	int Top = lua_gettop(L);
	x_0[i-1] = lua_tonumber(L, Top);
	lua_pop(L,1);	
  }

  luaL_checktype(L,2, LUA_TTABLE); // 1st argument must be a table (t)
  for (i=1; i<=n_params; i++){		
	lua_rawgeti(L, 2, i);  // push t
	int Top = lua_gettop(L);
	params[i-1] = lua_tonumber(L, Top);
	lua_pop(L,1);	
  }



  // Allocate memory (thread-safe)
  casadi_c_incref_id(id);
  int mem = casadi_c_checkout_id(id);

  /* Evaluate the function */
  arg[0] = x_0;
  arg[1] = params;
  res[0] = result;

  // Evaluation is thread-safe
  if (casadi_c_eval_id(id, arg, res, iw, w, mem)) return 1;

  // Release thread-local (not thread-safe)
  casadi_c_release_id(id, mem);



  printf("val result[1] = %f and res[2] = %f and res[3] = %f \n", result[0], result[1], result[2]);
  lua_newtable(L); //creating a lua table to pass the solution to lua
  for(i = 1; i<=n; i++){
  	
   	lua_pushnumber(L, result[i-1]);
   	lua_rawseti(L, -2, i);
   }

   /* Free memory (thread-safe) */
  casadi_c_decref_id(id);

  // Clear the last loaded Function(s) from the stack
  casadi_c_pop();

  return 1;
}

static const struct luaL_Reg conversionreg[] = {
 {"hello_world", hello_world},
 {"call_ocp", ocp_fun},
 {"callmp_rightarm", mp_rightarm},
 {"callmp_dualarm", mp_dualarm},
 {NULL, NULL}
};

int luaopen_motionplanner (lua_State *L) {
 luaL_register(L, "libmotionplanner", conversionreg); //Lua 5.1
  //luaL_newlib(L,conversionreg);
  return 1;
}





int main(){

//   // Variables
//   SX x = SX::sym("x", 2, 2);
//   SX y = SX::sym("y");

//   // Simple function
//   Function f("f", {x, y}, {x*y});

//   // Mode 1: Function::save
//   f.save("f.casadi");

//   // More simple functions
//   Function g("g", {x, y}, {sqrt(y)-1, sin(x)-y});
//   Function h("h", {x}, {y*y});

//   // Mode 2: FileSerializer (allows packing a list of Functions)
//   {
//     FileSerializer gh("gh.casadi",{{"debug", true}});
//     gh.pack(std::vector<Function>{g, h});
//   }

//   // Usage from C
   //double *sol;
   //usage_c(sol);

	lua_State *L;
  // hello_world(L);
	//mp_leftarm(L);

//   return 0;
 }