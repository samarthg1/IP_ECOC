import numpy as np
import math
import itertools
import time
import numpy as np
import random
import sys
from subprocess import PIPE, Popen
import os
from gurobipy import *
from collections import OrderedDict

random.seed(1)

def system(command):
    process = Popen(
        args=command,
        stdout=PIPE,
        shell=True
    )
    return process.communicate()[0]



def write_dimacs(k1,d1,no_nodes,tmp,filename_):

	K =k1
	d =d1

	h = "p EDGE "+str(no_nodes)+" "+str(tmp.shape[0])
	dimacs_filename = filename_  #"inf_pairs_bal/"+str(K)+"_"+str(d)+"_bal_graph.dimacs" 

	with open(dimacs_filename, "w") as f:
		# write the header
		f.write("p EDGE {} {}\n".format(no_nodes,tmp.shape[0]))
		# now write all edges
		for i in range(tmp.shape[0]):
			f.write("e {} {}\n".format(tmp[i][0], tmp[i][1]))

def read_cliques_from_coverfile(filename):
	with open(filename) as f:
	    content = f.readlines()
	# you may also want to remove whitespace characters like `\n` at the end of each line
	content = [[int(i) for i in (x.strip()).split(' ')] for x in content] 	
	#print(content)
	return content


def sample_col_indices_sorted(min_index,max_index,size):
	col_indices = set([random.randint(min_index, max_index) for i in range(size) ] )
	
	while len(col_indices) < size:
		col_indices.add(random.randint(min_index, max_index))
	col_indices = list(col_indices)
	col_indices.sort()
	col_indices = [ 1.0*i for i in col_indices]
	return col_indices

def clear_dirs():
	command = "rm cover_files_tmp/*"
	system(command)
	command = "rm inf_pairs_tmp/*"
	system(command)


def find_optimal_subset(book, L_, d_):

	K_ = book.shape[0]
	col_D =   (K - np.matmul(book.T,book)) 

	#print(col_D)
	#print("Time taken to col_D:", time.time() - t1)

	t1 = time.time()
	infeas_indices = np.argwhere( col_D < 2*d_) 
	#infeas_indices = np.argwhere( col_D < d)

	#print(infeas_indices,type(infeas_indices),infeas_indices.shape[0])
	infeas_indices =  infeas_indices[infeas_indices[:,0] < infeas_indices[:,1] ]   ## indexing starting from 0, not adding 1


	print("No of infeasible pairs:", infeas_indices.shape[0])
	#print(infeas_indices)
	print("Time taken to infeas:", time.time() - t1)
	
	cliques = []
	no_cliques = 0
	if infeas_indices.shape[0] > 0:
		clear_dirs()
		t1 = time.time()
		filename = "inf_pairs_tmp/"+str(K_)+"_"+str(d_)+"_rnd_graph.dimacs"
		write_dimacs(K_,d_,book.shape[1],infeas_indices,filename)
		print("Time taken to write dimacs:", time.time() - t1)

		t1 = time.time()
		terminal_output = "log_"+str(K_)+"_"+str(d_)
		#command = "java -Xmx31g  -jar ECC8.jar -g "+filename+" -o ./cover_files -f dimacs  > "+terminal_output
		command = "java -Xmx8g -jar ECC8.jar -g "+filename+" -o ./cover_files_tmp -f dimacs  > "+terminal_output
		print(command)
		system(command)
		print("Time taken to find cliques:", time.time() - t1)


		t1 = time.time()
		cover_filename = "cover_files_tmp/"+str(K)+"_"+str(d)+"_rnd_graph.dimacs-rand.EPSc.cover"
		cliques= read_cliques_from_coverfile(cover_filename)
		no_cliques = len(cliques)
		print("no of cliques:", no_cliques)
		print("Time taken to read cliques:", time.time() - t1)


	no_columns_ = book.shape[1]
	print("no of cols:",no_columns_)


	row_HammingCompare = np.zeros( (int(0.5*K_*(K_-1)), no_columns_) )


	t=0
	for i in range(K_-1):
		for j in range(i+1,K_):
			#print(t, i,j)
			row_HammingCompare[t,:] = 1*(book[i,:] != book[j,:])
			t=t+1


	m = Model('IP2')
	x_i = m.addVars(no_columns_ ,vtype=GRB.BINARY, name='x')

	m.addConstr( (x_i.sum() <= L_),name='no_of_columns' )

	tt = m.addVar(vtype=GRB.INTEGER,name='tt')


	for i in range( int(0.5*K_*(K_-1)) ):
	    m.addConstr( tt <= sum( [ row_HammingCompare[i,j]*x_i[j] for j in range(no_columns_)]) ) #>= tt )

	m.setObjective(tt, GRB.MAXIMIZE)


	m.Params.Seed = 100

	for i in range( no_cliques ):
		m.addConstr( sum( [ x_i[ j  ] for j in cliques[i] ] )  <= 1 ,name="clique_"+str(i) )

	m.update()
	#m.display()
	m.Params.Presolve = 0
	#m.Params.GomoryPasses = 2000000
	#m.Params.CoverCuts = 2
	#m.Params.ZeroHalfCuts = 2
	#m.Params.TimeLimit = 1000
	m.Params.method = 0
	#m.Params.Heuristics = 0.5
	m.optimize()
	print("Runtime:", m.Runtime)
	print("Gap:", m.MIPGap)
	#print("Obj:", m.objVal)
	#print("Bound:", m.objBound)
	#print("Status:", m.Status)
	print('------------------------------------------------------------------------')


soln_dict = OrderedDict()

### Creating folders to save files for edge-clique-covers
system("mkdir -p cover_files_tmp")
system("mkdir -p inf_pairs_tmp")

for k in range(10,18):
	soln_dict[k] = []
	
	for run_id in range(5):
		clear_dirs()	
	
		K = k
		L = 2*K
		d = int(math.floor(K*1.0/3))
		print("K:",K,"L:",L,"d:",d, "run_id:",run_id)
		no_columns = 2**(K-1) -1
	
			
		book_feas = np.ones((K,no_columns ))

		t1 = time.time()
		for i in range(2,K+1):
			for j in range(1,no_columns+1):
				book_feas[i-1,j-1] = int( 1*( math.ceil(j*1.0/(2**(K-i)))%2 == 0) )
	
		print("time taken to create book_feas:", time.time()-t1)
		print(book_feas,book_feas.shape)
		book_feas[ book_feas == 0 ] = -1

		book_feas = book_feas.astype('int8')

	
		t1 = time.time()

		col_D =   (K - np.matmul(book_feas.T,book_feas)) 

		#print(col_D)
		print("Time taken to col_D:", time.time() - t1)

		t1 = time.time()
		infeas_indices = np.argwhere( col_D < 2*d) 
		#infeas_indices = np.argwhere( col_D < d)

		#print(infeas_indices,type(infeas_indices),infeas_indices.shape[0])
		infeas_indices =  infeas_indices[infeas_indices[:,0] < infeas_indices[:,1] ]   ## indexing starting from 0, not adding 1


		print("No of infeasible pairs:", infeas_indices.shape)
		#print(infeas_indices)
		print("Time taken to infeas:", time.time() - t1)
	
		t1 = time.time()
		filename = "inf_pairs_tmp/"+str(K)+"_"+str(d)+"_rnd_graph.dimacs"
		write_dimacs(K,d,book_feas.shape[1],infeas_indices,filename)
		print("Time taken to write dimacs:", time.time() - t1)

		t1 = time.time()
		terminal_output = "log_"+str(K)+"_"+str(d)
		#command = "java -Xmx31g  -jar ECC8.jar -g "+filename+" -o ./cover_files -f dimacs  > "+terminal_output
		command = "java -Xmx32g -jar ECC8.jar -g "+filename+" -o ./cover_files_tmp -f dimacs  > "+terminal_output
		print(command)
		system(command)
		print("Time taken to find cliques:", time.time() - t1)


		t1 = time.time()
		cover_filename = "cover_files_tmp/"+str(K)+"_"+str(d)+"_rnd_graph.dimacs-rand.EPSc.cover"
		cliques= read_cliques_from_coverfile(cover_filename)
		no_cliques = len(cliques)
		print("no of cliques:", no_cliques)
		print("Time taken to read cliques:", time.time() - t1)


		no_columns_feas = book_feas.shape[1]
		print("no of feas cols:",no_columns_feas)


		row_HammingCompare_feas = np.zeros( (int(0.5*K*(K-1)), no_columns_feas) )


		t=0
		for i in range(K-1):
			for j in range(i+1,K):
				#print(t, i,j)
				row_HammingCompare_feas[t,:] = 1*(book_feas[i,:] != book_feas[j,:])
				t=t+1


		m = Model('IP2')
		x_i = m.addVars(no_columns_feas ,vtype=GRB.BINARY, name='x')

		m.addConstr( (x_i.sum() == L),name='no_of_columns' )

		tt = m.addVar(vtype=GRB.INTEGER,name='tt', lb=1)


		for i in range( int(0.5*K*(K-1)) ):
		    m.addConstr( tt <= sum( [ row_HammingCompare_feas[i,j]*x_i[j] for j in range(no_columns_feas)]) ) #>= tt )

		m.setObjective(tt, GRB.MAXIMIZE)


		m.Params.Seed = run_id



		for i in range( no_cliques ):
		 	m.addConstr( sum( [ x_i[ j  ] for j in cliques[i] ] )  <= 1 ,name="clique_"+str(i) )

		m.update()
		#m.display()
		m.Params.Presolve = 0
		#m.Params.GomoryPasses = 2000000
		#m.Params.CoverCuts = 2
		#m.Params.ZeroHalfCuts = 2
		m.Params.TimeLimit = 1800
		m.Params.method = 0
		#m.Params.Heuristics = 0.5
		m.optimize()
		print("Runtime:", m.Runtime)
		print("Gap:", m.MIPGap)
		print("Obj:", m.objVal)
		#print("Bound:", m.objBound)
		#print("Status:", m.Status)
		print('------------------------------------------------------------------------')

		soln_X = np.array( [ int(x_i[i].X) for i in range(no_columns_feas) ] )
	
		#print(soln_X)
	
		soln_indices = np.argwhere( soln_X == 1)
		#print(soln_indices, type(soln_indices))
		soln_indices = list(soln_indices[:,0])
		#print(soln_indices, type(soln_indices))

		soln_book = book_feas[:,soln_indices]
		#soln_book= soln_book.astype('int8')
		#print(soln_book)
		print("soln_book size:",soln_book.shape) 
		#np.save("feas_soln_book_"+str(K)+"_d_"+str(d),soln_book)
		'''
		d_h_row = 0.5*(soln_book.shape[1] - np.matmul(soln_book,soln_book.T))
		d_h_row = d_h_row.astype('int16')
		print(d_h_row)
		print('------------------------------------------------------------------------')
		d_h_row = np.triu(d_h_row)
		d_h_row_val= d_h_row[ d_h_row > 0 ]
		print('------------------------------------------------------------------------')
		print(d_h_row_val, type(d_h_row_val))
		print('------------------------------------------------------------------------')
		print(np.histogram(d_h_row_val) )
		'''
		#find_optimal_subset(soln_book, 10, d)

		soln_dict[k].append(m.objVal)

for key in soln_dict.keys():
	print(key, soln_dict[key],  "avg:", sum(soln_dict[key])*1.0/len(soln_dict[key])   )
