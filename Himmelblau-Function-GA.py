import numpy as np
import matplotlib.pyplot as plt
import random as rd


p_c = 1
p_m = 0.2
K = 3
pop = 160
gen = 60


# x and y, 13 for x and 13 for y

XY0 = np.array([0,1,0,0,0,1,0,0,1,0,1,1,1,0,1,1,1,0,0,1,0,1,1,0,1,1])
XY_Encoded_Before = np.array([0,1,0,0,0,1,0,0,1,0,1,1,1,0,1,1,1,0,0,1,0,1,1,0,1,1])

n_list = np.empty((0,len(XY0)))

for i in range(pop): # Shuffle the elements in the vector n times and store them
    rd.shuffle(XY0) # Baraja los valores.
    n_list = np.vstack((n_list,XY0)) # Añade cada array generado como una fila más.


# Calculate fitness value
def objective_value(array):  
    a_X = -6
    b_X = 6
    l_X = (len(array)//2) # Length of chromosome X
    
    a_Y = -6
    b_Y = 6
    l_Y = (len(array)//2) # Length of chromosome X
    
    Precision_X = (b_X-a_X)/((2**l_X)-1)
    Precision_Y = (b_Y-a_Y)/((2**l_Y)-1)
    
    # [0,1,0,0,0,1,0,0,1,0,1,1,1,0,1,1,1,0,0,1,0,1,1,0,1,1]
    
    z = 0
    t = 1
    X0_num_Sum = 0
    
    # Calculamos el fitness value o FENOTIPO
    for i in range(len(array)//2):
        X0_sum = array[-t]*(2**z) # Recorremos el array de derecha a izquierda por eso se usa -t
        X0_num_Sum = X0_num_Sum + X0_sum
        t = t+1
        z = z+1   
    
    p = 0
    u = 1 + (len(array)//2)
    Y0_num_Sum = 0
    
    for j in range(len(array)//2):
        Y0_sum = array[-u]*(2**p)
        Y0_num_Sum = Y0_num_Sum + Y0_sum
        u = u+1
        p = p+1
    
    
    Decoded_X = (X0_num_Sum*Precision_X)+a_X
    Decoded_Y = (Y0_num_Sum*Precision_Y)+a_Y
    
    #Calculamos la función objetivo
    OF_So_Far = ((Decoded_X**2)+Decoded_Y-11)**2+(Decoded_X+(Decoded_Y**2)-7)**2
    
    return Decoded_X,Decoded_Y,OF_So_Far
    


Final_Best_in_Generation_X = []
Final_Worse_in_Generation_X = []

# pnemos el +1 para guardar el fitness value
For_Plotting_the_Best = np.empty((0,len(XY0)+1))

# Ponemos el +2 para tener traza de la generación y el valor de coste
One_Final_Guy = np.empty((0,len(XY0)+2))
One_Finel_Guy_Final = []

Min_for_all_Generations_for_Mut_1 = np.empty((0,len(XY0)+1))
Min_for_all_Generations_for_Mut_2 = np.empty((0,len(XY0)+1))

Min_for_all_Generations_for_Mut_1_1 = np.empty((0,len(XY0)+2))
Min_for_all_Generations_for_Mut_2_2 = np.empty((0,len(XY0)+2))

Min_for_all_Generations_for_Mut_1_1_1 = np.empty((0,len(XY0)+2))
Min_for_all_Generations_for_Mut_2_2_2 = np.empty((0,len(XY0)+2))



Generation = 1

for i in range(gen):
    
    New_Population = np.empty((0,len(XY0))) # Saving the new generation
    
    All_in_Generation_X_1 = np.empty((0,len(XY0)+1))
    All_in_Generation_X_2 = np.empty((0,len(XY0)+1))
    
    Min_in_Generation_X_1 = []
    Min_in_Generation_X_2 = []
    
    Save_Best_in_Generation_X = np.empty((0,len(XY0)+1))
    Final_Best_in_Genration_X = []
    Final_Worst_in_Genration_X = []
    
    print("--> Generation: #", Generation)
        
    Family = 1
    
    for j in range(int(pop/2)): # Range N/2 porque elegimos a dos padres
        
        print("--> Family: #", Family)
            
        # Tournament Selection
        
        Parents = np.empty((0,len(XY0)))
        
        for i in range(2):
            
            Battle_Troops = []
            
            Warrior_1_index = np.random.randint(0,len(n_list)) # e.g. 5
            Warrior_2_index = np.random.randint(0,len(n_list))
            Warrior_3_index = np.random.randint(0,len(n_list))
           
            # Nos aseguramos que cogemos 3 diferentes
            while Warrior_1_index==Warrior_2_index:
                Warrior_1_index = np.random.randint(0,len(n_list))
            while Warrior_2_index==Warrior_3_index:
                Warrior_3_index = np.random.randint(0,len(n_list))
            while Warrior_1_index==Warrior_3_index:
                Warrior_3_index = np.random.randint(0,len(n_list))
                
            Warrior_1 = n_list[Warrior_1_index] # n_list[5]
            Warrior_2 = n_list[Warrior_2_index]
            Warrior_3 = n_list[Warrior_3_index]
            
            Battle_Troops = [Warrior_1,Warrior_2,Warrior_3]
            
            ### Warrior 1
            
            OF_So_Far_W1 = objective_value(Warrior_1)[2]
            
            Prize_Warrior_1 = OF_So_Far_W1
            
            
            ### Warrior 2
            
            OF_So_Far_W2 = objective_value(Warrior_2)[2]
            
            Prize_Warrior_2 = OF_So_Far_W2
            
            ### Warrior 3
            
            OF_So_Far_W3 = objective_value(Warrior_3)[2]
            
            Prize_Warrior_3 = OF_So_Far_W3
            
            if Prize_Warrior_1 == min(Prize_Warrior_1,Prize_Warrior_2,Prize_Warrior_3):
                Winner = Warrior_1
                Winner_str = "Warrior_1"
                Prize = Prize_Warrior_1
            elif Prize_Warrior_2 == min(Prize_Warrior_1,Prize_Warrior_2,Prize_Warrior_3):
                Winner = Warrior_2
                Winner_str = "Warrior_2"
                Prize = Prize_Warrior_2
            else:
                Winner = Warrior_3
                Winner_str = "Warrior_3"
                Prize = Prize_Warrior_3
            
            Parents = np.vstack((Parents,Winner))
            
            
        Parent_1 = Parents[0,:]
        Parent_2 = Parents[1,:]
        
        
        
        
        
        # Crossover
        
        Child_1 = np.empty((0,len(XY0)))
        Child_2 = np.empty((0,len(XY0)))
        
        # Where to crossover
        
        Ran_CO_1 = np.random.rand()
        
        
        if Ran_CO_1 < p_c:
            # Si por probabilidad hay crossover
            
            Cr_1 = np.random.randint(0,len(XY0))
            Cr_2 = np.random.randint(0,len(XY0))
            
            while Cr_1 == Cr_2:
                Cr_2 = np.random.randint(0,len(XY0))
                
            if Cr_1 < Cr_2:
                
                # [0,1,0,0,0,1,0,0,1,0,1,1,1,0,1,1,1,0,0,1,0,1,1,0,1,1]
                
                
                Mid_Seg_1 = Parent_1[Cr_1:Cr_2+1]
                Mid_Seg_2 = Parent_2[Cr_1:Cr_2+1]
                
                First_Seg_1 = Parent_1[:Cr_1]
                Last_Seg_1 = Parent_1[Cr_2+1:]
                
                First_Seg_2 = Parent_2[:Cr_1]
                Last_Seg_2 = Parent_2[Cr_2+1:]
            
                Child_1 = np.concatenate((First_Seg_1,Mid_Seg_2,Last_Seg_1))
                Child_2 = np.concatenate((First_Seg_2,Mid_Seg_1,Last_Seg_2))
                
            else:
                
                Mid_Seg_1 = Parent_1[Cr_2:Cr_1+1]
                Mid_Seg_2 = Parent_2[Cr_2:Cr_1+1]
                
                First_Seg_1 = Parent_1[:Cr_2]
                Last_Seg_1 = Parent_1[Cr_1+1:]
                
                First_Seg_2 = Parent_2[:Cr_2]
                Last_Seg_2 = Parent_2[Cr_1+1:]
            
                Child_1 = np.concatenate((First_Seg_1,Mid_Seg_2,Last_Seg_1))
                Child_2 = np.concatenate((First_Seg_2,Mid_Seg_1,Last_Seg_2))
            
        else:
            
            # Si por probablilidad no hay crossover
            Child_1 = Parent_1
            Child_2 = Parent_2
          
        
        # Mutation
        
        # Mutated_Child_1
        
        Mutated_Child_1 = []
        
        # [0,1,0,0,0,1,0,0,1,0,1,1,1,0,1,1,1,0,0,1,0,1,1,0,1,1]
        
        t = 0
        
        for i in Child_1:
            rand_Mut_1 = np.random.rand() # prob. to mutate
            
            if rand_Mut_1 < p_m:
                if Child_1[t] == 0:
                    Child_1[t] = 1
                else:
                    Child_1[t] = 0
                t = t+1
            
                Mutated_Child_1 = Child_1
            
            else:
                Mutated_Child_1 = Child_1
            
        
        # Mutated_Child_2
        
        Mutated_Child_2 = []
        
        
        t = 0
        
        for i in Child_2:
            rand_Mut_2 = np.random.rand() # prob. to mutate
            
            if rand_Mut_2 < p_m:
                if Child_2[t] == 0:
                    Child_2[t] = 1
                else:
                    Child_2[t] = 0
                t = t+1
            
                Mutated_Child_2 = Child_2
            
            else:
                Mutated_Child_2 = Child_2
            
        
        
        
        # For Mutated_1
        OF_So_Far_MC_1 = objective_value(Mutated_Child_1)[2]
        
        
        # For Mutated_2
        OF_So_Far_MC_2 = objective_value(Mutated_Child_2)[2]
        
        
        print()
        print("FV at Mutated Child #1 at Generation #",Generation,":",OF_So_Far_MC_1)
        print("FV at Mutated Child #2 at Generation #",Generation,":",OF_So_Far_MC_2)

        # Pasamos de array horizontal a vertical para tener má facil manejo
        All_in_Generation_X_1_1_temp = Mutated_Child_1[np.newaxis] 
        All_in_Generation_X_1_1 = np.column_stack((OF_So_Far_MC_1,All_in_Generation_X_1_1_temp))

        All_in_Generation_X_2_1_temp = Mutated_Child_2[np.newaxis]
        All_in_Generation_X_2_1 = np.column_stack((OF_So_Far_MC_2,All_in_Generation_X_2_1_temp))
        
        All_in_Generation_X_1 = np.vstack((All_in_Generation_X_1,All_in_Generation_X_1_1))
        All_in_Generation_X_2 = np.vstack((All_in_Generation_X_2,All_in_Generation_X_2_1))
        
        Save_Best_in_Generation_X = np.vstack((All_in_Generation_X_1,All_in_Generation_X_2))
        
        New_Population = np.vstack((New_Population, Mutated_Child_1, Mutated_Child_2))
        
        
        t = 0
        r_1 = []
        
        for i in All_in_Generation_X_1:
            if(All_in_Generation_X_1[t,:1]) <= min(All_in_Generation_X_1[:,:1]):
                r_1 = All_in_Generation_X_1[t,:]
            t = t+1


        Min_in_Generation_X_1 = r_1[np.newaxis]
        
        t = 0
        r_2 = []
        
        for i in All_in_Generation_X_2:
            if(All_in_Generation_X_2[t,:1]) <= min(All_in_Generation_X_2[:,:1]):
                r_2 = All_in_Generation_X_2[t,:]
            t = t+1


        Min_in_Generation_X_2 = r_2[np.newaxis]
        
        Family = Family+1
        
        
        
    t = 0
    r_final = []
    
    for i in Save_Best_in_Generation_X:
        
        if(Save_Best_in_Generation_X[t,:1]) <= min(Save_Best_in_Generation_X[:,:1]):
            r_final = Save_Best_in_Generation_X[t,]
        t = t+1
        
    Final_Best_in_Genration_X = r_final[np.newaxis]
        
    For_Plotting_the_Best = np.vstack((For_Plotting_the_Best,Final_Best_in_Genration_X))
    
    
    t = 0
    r_2_final = []
    
    for i in Save_Best_in_Generation_X:
        
        if(Save_Best_in_Generation_X[t,:1]) >= max(Save_Best_in_Generation_X[:,:1]):
            r_2_final = Save_Best_in_Generation_X[t,]
        t = t+1
        
    Final_Worst_in_Genration_X = r_2_final[np.newaxis]
    
    
    # Elitism
    
    Darwin_Guy = Final_Best_in_Genration_X[:]
    Not_So_Darwin_Guy = Final_Worst_in_Genration_X[:]
    
    Darwin_Guy = Darwin_Guy[0:,1:].tolist()
    Not_So_Darwin_Guy = Not_So_Darwin_Guy[0:,1:].tolist()
    
    
    Best_1 = np.where((New_Population==Darwin_Guy).all(axis=1))
    Worst_1 = np.where((New_Population==Not_So_Darwin_Guy).all(axis=1))
        
        
    New_Population[Worst_1] = Darwin_Guy
    
    n_list = New_Population
    
    
    Min_for_all_Generations_for_Mut_1 = np.vstack((Min_for_all_Generations_for_Mut_1,Min_in_Generation_X_1))
            
    Min_for_all_Generations_for_Mut_2 = np.vstack((Min_for_all_Generations_for_Mut_2,Min_in_Generation_X_2))       
            
    
    
    Min_for_all_Generations_for_Mut_1_1 = np.insert(Min_in_Generation_X_1,0,Generation) 
    Min_for_all_Generations_for_Mut_2_2 = np.insert(Min_in_Generation_X_2,0,Generation)

    
    Min_for_all_Generations_for_Mut_1_1_1 = np.vstack((Min_for_all_Generations_for_Mut_1_1_1,Min_for_all_Generations_for_Mut_1_1))
    Min_for_all_Generations_for_Mut_2_2_2 = np.vstack((Min_for_all_Generations_for_Mut_2_2_2,Min_for_all_Generations_for_Mut_2_2))      
            
    Generation = Generation+1       
            
    
One_Final_Guy = np.vstack((Min_for_all_Generations_for_Mut_1_1_1,Min_for_all_Generations_for_Mut_2_2_2))

t = 0
Final_Here = []

for i in One_Final_Guy:
    if(One_Final_Guy[t,1]) <= min (One_Final_Guy[:,1]):
        Final_Here = One_Final_Guy[t,:]
    t = t+1

One_Final_Guy_Final = Final_Here[np.newaxis]

print()
print("Min in all Generation:",One_Final_Guy_Final.tolist())
print("Min in Cost:",One_Final_Guy_Final[:,1])
print()

A = (One_Final_Guy_Final[:,1])

plt.plot(For_Plotting_the_Best[:,0])
plt.axhline(y=A,color='r',linestyle='--')
plt.title("Z Reached Through Generations",fontsize=20,fontweight='bold')
plt.xlabel("Generations",fontsize=18,fontweight='bold')
plt.ylabel("Z",fontsize=18,fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')

xyz = (Generation/4,A)
xyzz = (Generation/3.7,A+0.5)

plt.annotate("Minimum Cost Reached at: %0.3f" % A, xy=xyz, xytext=xyzz,
             arrowprops=dict(facecolor='black',shrink=1,width=1,headwidth=5),
             fontsize=12,fontweight='bold')

plt.show()

# [Generation number, ftness value, 0,1,0,0,0,1,0,0,1,0,1,1,1,0,1,1,1,0,0,1,0,1,1,0,1,1]

XY0_Encoded_After = Final_Here[2:]


# DECODING
# DECODING
# DECODING


Final_Solution = objective_value(XY0_Encoded_After)


print()
print()
print("X After:",Final_Solution[0])
print("Y After:",Final_Solution[1])
print("OF After:",Final_Solution[2])
































