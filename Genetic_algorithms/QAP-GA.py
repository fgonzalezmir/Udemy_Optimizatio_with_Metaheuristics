import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random as rd

Dist = pd.DataFrame([[0,1,2,3,1,2,3,4],[1,0,1,2,2,1,2,3],[2,1,0,1,3,2,1,2],
                      [3,2,1,0,4,3,2,1],[1,2,3,4,0,1,2,3],[2,1,2,3,1,0,1,2],
                      [3,2,1,2,2,1,0,1],[4,3,2,1,3,2,1,0]],
                    columns=["A","B","C","D","E","F","G","H"],
                    index=["A","B","C","D","E","F","G","H"])

Flow = pd.DataFrame([[0,5,2,4,1,0,0,6],[5,0,3,0,2,2,2,0],[2,3,0,0,0,0,0,5],
                      [4,0,0,0,5,2,2,10],[1,2,0,5,0,10,0,0],[0,2,0,2,10,0,5,1],
                      [0,2,0,2,0,5,0,10],[6,0,5,10,0,1,10,0]],
                    columns=["A","B","C","D","E","F","G","H"],
                    index=["A","B","C","D","E","F","G","H"])


Initial_Solution = ["D","A","C","B","G","E","F","H"]


# Make a dataframe of the initial solution
New_Dist_DF = Dist.reindex(columns=Initial_Solution, index=Initial_Solution)
New_Dist_Arr = np.array(New_Dist_DF)
# Make a dataframe of the cost of the initial solution
Objfun1_start = pd.DataFrame(New_Dist_Arr*Flow)
Objfun1_start_Arr = np.array(Objfun1_start)
sum_start_int = sum(sum(Objfun1_start_Arr))
print(sum_start_int)


# Las lineas de justo arriba las pasamos a una función para reutilizaras
def distance(array):
    New_Dist_DF = Dist.reindex(columns=array, index=array)
    New_Dist_Arr = np.array(New_Dist_DF)
    # Make a dataframe of the cost of the initial solution
    Objfun1_start = pd.DataFrame(New_Dist_Arr*Flow)
    Objfun1_start_Arr = np.array(Objfun1_start)
    sum_start_int = sum(sum(Objfun1_start_Arr))
    return(sum_start_int)


### VARIABLES ###
### VARIABLES ###
p_c = 1 # Probability of crossover
p_m = 0.3 # Probability of mutation
K = 3 # For Tournament selection
pop = 100 # Population per generation
gen = 30 # Number of generations
### VARIABLES ###
### VARIABLES ###


X0 = Initial_Solution[:]


# 1- Randomly generate n solutions, for generation #1

n_list = np.empty((0,len(X0)))

# Generamos la población incial
for i in range(int(pop)): # Shuffles the elements in the vector n times and stores them
    rnd_sol_1 = rd.sample(Initial_Solution,len(X0))
    n_list = np.vstack((n_list,rnd_sol_1))
    


Final_Best_in_Generation_X = []
Worst_Best_in_Generation_X = []

For_Plotting_the_Best = np.empty((0,len(X0)+1))

One_Final_Guy = np.empty((0,len(X0)+2))
One_Final_Guy_Final = []

# Teneos un +1 para almacenar e fitness value.
Min_for_all_Generations_for_Mut_1 = np.empty((0,len(X0)+1))
Min_for_all_Generations_for_Mut_2 = np.empty((0,len(X0)+1))

# Tenemos un +2 para guardar el fitness value y la generación
Min_for_all_Generations_for_Mut_1_1 = np.empty((0,len(X0)+2))
Min_for_all_Generations_for_Mut_2_2 = np.empty((0,len(X0)+2))

Min_for_all_Generations_for_Mut_1_1_1 = np.empty((0,len(X0)+2))
Min_for_all_Generations_for_Mut_2_2_2 = np.empty((0,len(X0)+2))


Generation = 1 


for i in range(gen):
    
    
    New_Population = np.empty((0,len(X0))) # Saving the new generation
    
    All_in_Generation_X_1 = np.empty((0,len(X0)+1))
    All_in_Generation_X_2 = np.empty((0,len(X0)+1))
    
    Min_in_Generation_X_1 = []
    Min_in_Generation_X_2 = []
    
    Save_Best_in_Generation_X = np.empty((0,len(X0)+1))
    Final_Best_in_Generation_X = []
    Worst_Best_in_Generation_X = []
    
    
    print()
    print("--> GENERATION: #",Generation)
    
    Family = 1
    
    for j in range(int(pop/2)): # range(int(pop/2))
        
        print()
        print("--> FAMILY: #",Family)
          
        
        # Tournament Selection
        # Tournament Selection
        # Tournament Selection
        
        Parents = np.empty((0,len(X0)))
        
        for i in range(2):
            
            Battle_Troops = []
            
            Warrior_1_index = np.random.randint(0,len(n_list))
            Warrior_2_index = np.random.randint(0,len(n_list))
            Warrior_3_index = np.random.randint(0,len(n_list))
            
            while Warrior_1_index == Warrior_2_index:
                Warrior_1_index = np.random.randint(0,len(n_list))
            while Warrior_2_index == Warrior_3_index:
                    Warrior_3_index = np.random.randint(0,len(n_list))
            while Warrior_1_index == Warrior_3_index:
                    Warrior_3_index = np.random.randint(0,len(n_list))
            
            Warrior_1 = n_list[Warrior_1_index,:]
            Warrior_2 = n_list[Warrior_2_index,:]
            Warrior_3 = n_list[Warrior_3_index,:]
            
            Battle_Troops = [Warrior_1,Warrior_2,Warrior_3]
            
            
            # For Warrior #1
            Prize_Warrior_1 = distance(Warrior_1) 
            
            
            # For Warrior #2
            Prize_Warrior_2 = distance(Warrior_2)
            
            
            # For Warrior #3
            Prize_Warrior_3 = distance(Warrior_3)
            
            
            
            
            if Prize_Warrior_1 == min(Prize_Warrior_1,Prize_Warrior_2,Prize_Warrior_3):
                Winner = Warrior_1
            elif Prize_Warrior_2 == min(Prize_Warrior_1,Prize_Warrior_2,Prize_Warrior_3):
                Winner = Warrior_2
            else:
                Winner = Warrior_3
            
        
            Parents = np.vstack((Parents,Winner))
        
        
        # Ya tenemos los dos padres seleccionados por torneo
        Parent_1 = Parents[0]
        Parent_2 = Parents[1]
        
        
        Child_1 = np.empty((0,len(X0)))
        Child_2 = np.empty((0,len(X0)))
        
        
        # Where to crossover
        
        Ran_CO_1 = np.random.rand()
        
        if Ran_CO_1 < p_c:
            
            # Choose two random numbers to crossover with their locations
            Cr_1 = np.random.randint(0,len(X0))
            Cr_2 = np.random.randint(0,len(X0))
            
            while Cr_1 == Cr_2:
                Cr_2 = np.random.randint(0,len(X0))
            
            if Cr_1 < Cr_2:
            
                Cr_2 = Cr_2 + 1
                
                
                New_Dep_1 = Parent_1[Cr_1:Cr_2] # Mid seg. of parent #1
                
                New_Dep_2 = Parent_2[Cr_1:Cr_2] # Mid seg. of parent #2
                
                First_Seg_1 = Parent_1[:Cr_1] # First seg. of parent #1
                
                First_Seg_2 = Parent_2[:Cr_1] # First seg. of parent #2
        
                Temp_First_Seg_1_1 = [] # Temporay for first seg.
                Temp_Second_Seg_2_2 = [] # Temporay for second seg.
                
                Temp_First_Seg_3_3 = [] # Temporay for first seg.
                Temp_Second_Seg_4_4 = [] # Temporay for second seg.
                
                
                # No se puede tener el mismo departamento repetido.
                for i in First_Seg_2: # For i in all the elements of the first segment of parent #2
                    if i not in New_Dep_1: # If they aren't in seg. parent #1
                        Temp_First_Seg_1_1 = np.append(Temp_First_Seg_1_1,i) # Append them
                
                Temp_New_Vector_1 = np.concatenate((Temp_First_Seg_1_1,New_Dep_1)) # Add it next to the mid seg.
                
                for i in Parent_2: # For parent #2
                    if i not in Temp_New_Vector_1: # If not in what is made so far ^^
                        Temp_Second_Seg_2_2 = np.append(Temp_Second_Seg_2_2,i) # Append it
                
                Child_1 = np.concatenate((Temp_First_Seg_1_1,New_Dep_1,Temp_Second_Seg_2_2)) # Now you can make the child from the elements
                
                for i in First_Seg_1: # Do the same for child #2
                    if i not in New_Dep_2:
                        Temp_First_Seg_3_3 = np.append(Temp_First_Seg_3_3,i)
                
                Temp_New_Vector_2 = np.concatenate((Temp_First_Seg_3_3,New_Dep_2))
        
                for i in Parent_1:
                    if i not in Temp_New_Vector_2:
                        Temp_Second_Seg_4_4 = np.append(Temp_Second_Seg_4_4,i)
                
                Child_2 = np.concatenate((Temp_First_Seg_3_3,New_Dep_2,Temp_Second_Seg_4_4))
    
            else: # The same in reverse of Cr_1 and Cr_2
            
                Cr_1 = Cr_1 + 1
                
                New_Dep_1 = Parent_1[Cr_2:Cr_1]
                
                New_Dep_2 = Parent_2[Cr_2:Cr_1]
                
                First_Seg_1 = Parent_1[:Cr_2]
                
                First_Seg_2 = Parent_2[:Cr_2]
        
                Temp_First_Seg_1_1 = []
                Temp_Second_Seg_2_2 = []
                
                Temp_First_Seg_3_3 = []
                Temp_Second_Seg_4_4 = []
                
                for i in First_Seg_2:
                    if i not in New_Dep_1:
                        Temp_First_Seg_1_1 = np.append(Temp_First_Seg_1_1,i)
                
                Temp_New_Vector_1 = np.concatenate((Temp_First_Seg_1_1,New_Dep_1))
                
                for i in Parent_2:
                    if i not in Temp_New_Vector_1:
                        Temp_Second_Seg_2_2 = np.append(Temp_Second_Seg_2_2,i)
                
                Child_1 = np.concatenate((Temp_First_Seg_1_1,New_Dep_1,Temp_Second_Seg_2_2))
                
                for i in First_Seg_1:
                    if i not in New_Dep_2:
                        Temp_First_Seg_3_3 = np.append(Temp_First_Seg_3_3,i)
                
                Temp_New_Vector_2 = np.concatenate((Temp_First_Seg_3_3,New_Dep_2))
        
                for i in Parent_1:
                    if i not in Temp_New_Vector_2:
                        Temp_Second_Seg_4_4 = np.append(Temp_Second_Seg_4_4,i)
                
                Child_2 = np.concatenate((Temp_First_Seg_3_3,New_Dep_2,Temp_Second_Seg_4_4))
                 
        
        else: # If random number was above p_c
            
            Child_1 = Parent_1
            Child_2 = Parent_2
            
    
        # Mutation Child #1
        # Mutation Child #1
        # Mutation Child #1
        
        Mutated_Child_1 = []

        
        Ran_Mut_1 = np.random.rand() # Probablity to Mutate
        Ran_Mut_2 = np.random.randint(0,len(X0))
        Ran_Mut_3 = np.random.randint(0,len(X0))
        
        A1 = Ran_Mut_2
        A2 = Ran_Mut_3
        
        while A1 == A2:
            A2 = np.random.randint(0,len(X0))
        
        if Ran_Mut_1 < p_m: # If probablity to mutate is less than p_m, then mutate
            if A1 < A2:
                M_Child_1_Pos_1 = Child_1[A1] # Take the index
                M_Child_1_Pos_2 = Child_1[A2] # Take the index
                A2 = A2+1
                Rev_1 = Child_1[:] # copy of child #1
                Rev_2 = list(reversed(Child_1[A1:A2])) # reverse the order
                t = 0
                for i in range(A1,A2):
                    Rev_1[i] = Rev_2[t] # The reversed will become instead of the original
                    t = t+1
                
                Mutated_Child_1 = Rev_1
            
            else:
                M_Child_1_Pos_1 = Child_1[A2]
                M_Child_1_Pos_2 = Child_1[A1]
                A1 = A1+1
                Rev_1 = Child_1[:]
                Rev_2 = list(reversed(Child_1[A2:A1]))
                t = 0
                for i in range(A2,A1):
                    Rev_1[i] = Rev_2[t]
                    t = t+1
                
                Mutated_Child_1 = Rev_1
        else:
            Mutated_Child_1 = Child_1
        
        
        
        Mutated_Child_2 = []

        
        Ran_Mut_1 = np.random.rand() # Probablity to Mutate
        Ran_Mut_2 = np.random.randint(0,len(X0))
        Ran_Mut_3 = np.random.randint(0,len(X0))
        
        A1 = Ran_Mut_2
        A2 = Ran_Mut_3
        
        while A1 == A2:
            A2 = np.random.randint(0,len(X0))
        
        if Ran_Mut_1 < p_m: # If probablity to mutate is less than p_m, then mutate
            if A1 < A2:
                M_Child_1_Pos_1 = Child_2[A1]
                M_Child_1_Pos_2 = Child_2[A2]
                A2 = A2+1
                Rev_1 = Child_2[:]
                Rev_2 = list(reversed(Child_2[A1:A2]))
                t = 0
                for i in range(A1,A2):
                    Rev_1[i] = Rev_2[t]
                    t = t+1
                
                Mutated_Child_2 = Rev_1
            
            else:
                M_Child_1_Pos_1 = Child_2[A2]
                M_Child_1_Pos_2 = Child_2[A1]
                A1 = A1+1
                Rev_1 = Child_2[:]
                Rev_2 = list(reversed(Child_2[A2:A1]))
                t = 0
                for i in range(A2,A1):
                    Rev_1[i] = Rev_2[t]
                    t = t+1
                
                Mutated_Child_2 = Rev_1
        else:
            Mutated_Child_2 = Child_2
        
        
        
        
        Total_Cost_Mut_1 = distance(Mutated_Child_1) 
        
        Total_Cost_Mut_2 = distance(Mutated_Child_2) 
        
        
        print()
        print("FV at Mutated Child #1 at Gen #",Generation,":", Total_Cost_Mut_1)
        print("FV at Mutated Child #2 at Gen #",Generation,":", Total_Cost_Mut_2)
        
        
        
        All_in_Generation_X_1_1_temp = Mutated_Child_1[np.newaxis]
        All_in_Generation_X_1_1 = np.column_stack((Total_Cost_Mut_1, All_in_Generation_X_1_1_temp))
        
        All_in_Generation_X_2_1_temp = Mutated_Child_2[np.newaxis]
        All_in_Generation_X_2_1 = np.column_stack((Total_Cost_Mut_2, All_in_Generation_X_2_1_temp))
        
        All_in_Generation_X_1 = np.vstack((All_in_Generation_X_1,All_in_Generation_X_1_1))
        All_in_Generation_X_2 = np.vstack((All_in_Generation_X_2,All_in_Generation_X_2_1))
        
        
        Save_Best_in_Generation_X = np.vstack((All_in_Generation_X_1,All_in_Generation_X_2))
        
        
        New_Population = np.vstack((New_Population,Mutated_Child_1,Mutated_Child_2))
        
        t = 0
        
        R_1 = []
        for i in All_in_Generation_X_1:
            
            if (All_in_Generation_X_1[t,:1]) <= min(All_in_Generation_X_1[:,:1]):
                R_1 = All_in_Generation_X_1[t,:]
            t = t+1
              
        Min_in_Generation_X_1 = R_1[np.newaxis]
        
        
        t = 0
        R_2 = []
        for i in All_in_Generation_X_2:
            
            if (All_in_Generation_X_2[t,:1]) <= min(All_in_Generation_X_2[:,:1]):
                R_2 = All_in_Generation_X_2[t,:]
            t = t+1
                
        Min_in_Generation_X_2 = R_2[np.newaxis]
        
        
        Family = Family+1
    
    t = 0
    R_Final = []
    
    for i in Save_Best_in_Generation_X:
        
        if (Save_Best_in_Generation_X[t,:1]) <= min(Save_Best_in_Generation_X[:,:1]):
            R_Final = Save_Best_in_Generation_X[t,:]
        t = t+1
    
    Final_Best_in_Generation_X = R_Final[np.newaxis]
    
    
    
    For_Plotting_the_Best = np.vstack((For_Plotting_the_Best,Final_Best_in_Generation_X))
    
    t = 0
    R_22_Final = []
    
    for i in Save_Best_in_Generation_X:
        
        if (Save_Best_in_Generation_X[t,:1]) >= max(Save_Best_in_Generation_X[:,:1]):
            R_22_Final = Save_Best_in_Generation_X[t,:]
        t = t+1
    
    Worst_Best_in_Generation_X = R_22_Final[np.newaxis]
    
    
    
    
    # Elitism, the best in the generation lives
    # Elitism, the best in the generation lives
    # Elitism, the best in the generation lives
    
    Darwin_Guy = Final_Best_in_Generation_X[:]
    Not_So_Darwin_Guy = Worst_Best_in_Generation_X[:]
    
    Darwin_Guy = Darwin_Guy[0:,1:].tolist()
    Not_So_Darwin_Guy = Not_So_Darwin_Guy[0:,1:].tolist()
    
    
    Best_1 = np.where((New_Population == Darwin_Guy).all(axis=1))
    Worst_1 = np.where((New_Population == Not_So_Darwin_Guy).all(axis=1))
    
    
    New_Population[Worst_1] = Darwin_Guy
    
    
    n_list = New_Population
    

    Min_for_all_Generations_for_Mut_1 = np.vstack((Min_for_all_Generations_for_Mut_1,Min_in_Generation_X_1))
    Min_for_all_Generations_for_Mut_2 = np.vstack((Min_for_all_Generations_for_Mut_2,Min_in_Generation_X_2))
    
    Min_for_all_Generations_for_Mut_1_1 = np.insert(Min_in_Generation_X_1, 0, Generation)
    Min_for_all_Generations_for_Mut_2_2 = np.insert(Min_in_Generation_X_2, 0, Generation)
    
    Min_for_all_Generations_for_Mut_1_1_1 = np.vstack((Min_for_all_Generations_for_Mut_1_1_1,Min_for_all_Generations_for_Mut_1_1))
    Min_for_all_Generations_for_Mut_2_2_2 = np.vstack((Min_for_all_Generations_for_Mut_2_2_2,Min_for_all_Generations_for_Mut_2_2))
    
    Generation = Generation+1
    



One_Final_Guy = np.vstack((Min_for_all_Generations_for_Mut_1_1_1,Min_for_all_Generations_for_Mut_2_2_2))
    
t = 0
Final_Here = []
for i in One_Final_Guy:
    
    if (One_Final_Guy[t,1]) <= min(One_Final_Guy[:,1]):
        Final_2 = []
        Final_2 = [One_Final_Guy[t,1]]
        Final_Here = One_Final_Guy[t,:]
    t = t+1
        
A_2_Final = min(One_Final_Guy[:,1])

One_Final_Guy_Final = Final_Here[np.newaxis]

print()
print("Min in all Generations:",One_Final_Guy_Final)

print("The Lowest Cost is:",One_Final_Guy_Final[:,1])

Look = (One_Final_Guy_Final[:,1]).tolist()
Look = float(Look[0])
Look = int(Look)

plt.plot(For_Plotting_the_Best[:,0])
plt.axhline(y=Look,color="r",linestyle='--')
plt.title("Cost Reached Through Generations",fontsize=20,fontweight='bold')
plt.xlabel("Generations",fontsize=18,fontweight='bold')
plt.ylabel("Cost (Flow * Distance)",fontsize=18,fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
xyz=(Generation/2, Look)
xyzz = (Generation/1.95, 1)
plt.annotate("Minimum Reached at: %s" % Look, xy=xyz, xytext=xyzz,
             arrowprops=dict(facecolor='black', shrink=0.001,width=1,headwidth=5),
             fontsize=12,fontweight='bold')
plt.show()

print()
print("Initial Solution:",X0)
print("Final Solution:",One_Final_Guy_Final[:,2:])
print("The Lowest Cost is:",One_Final_Guy_Final[:,1])
print("At Generation:",One_Final_Guy_Final[:,0])
print()
print("### METHODS ###")
print("# Selection Method = Tournament Selection")
print("# Crossover = C1 (order) but 2-point selection")
print("# Mutation = #1- Inverse")
print("# Other = Elitism")
print("### METHODS ###")
print()
print("### VARIABLES ###")
print("p_c = %s" % p_c)
print("p_m = %s" % p_m)
print("K = %s" % K)
print("pop = %s" % pop)
print("gen = %s" % gen)
print("### VARIABLES ###")
            
            
            
            
            
            
            