import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random as rd

# Es igual que la codificación de un Algoritmo Genetico con el añadido de la 
# función de penalización.

#Restricciones:
#        -5 < X1 < 5
#        -5 < x2 < 5


### VARIABLES ###
### VARIABLES ###
Seed = 0 # Seed for the random number generator
p_c = 1 # Probability of crossover
p_m = 0.2 # Probability of mutation
K = 3 # For Tournament selection
pop = 30 # Population per generation
gen = 30 # Number of generations
### VARIABLES ###
### VARIABLES ###


# Where the first 13 represent Y and the second 13 represent X
XY0 = np.array([0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1,
                1, 0, 0, 0, 1, 1, 1, 1, 1]) # Initial solution

Init_Sol = np.array([0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1,
                1, 0, 0, 0, 1, 1, 1, 1, 1]) # Initial solution

n_list = np.empty((0,len(XY0)))

# Generamos la población inicial
for i in range(pop): # Shuffles the elements in the vector n times and stores them
    rd.shuffle(XY0)
    n_list = np.vstack((n_list,XY0))

# Calculating fitness value

a_X = -5 # Lower bound of X
b_X = 5 # Upper bound of X
l_X = (len(XY0)//2) # Length of Chrom. X

a_Y = -5 # Lower bound of Y
b_Y = 5 # Upper bound of Y
l_Y = (len(XY0)//2) # Length of Chrom. Y


Precision_X = (b_X - a_X)/((2**l_X)-1)

Precision_Y = (b_Y - a_Y)/((2**l_Y)-1)

z = 0
t = 1
X0_num_Sum = 0

# Decodificamos X e Y 

for i in range(len(XY0)//2):
    X0_num = XY0[-t]*(2**z)
    X0_num_Sum += X0_num
    t = t+1
    z = z+1


p = 0
u = 1 + (len(XY0)//2)
Y0_num_Sum = 0

for j in range(len(XY0)//2):
    Y0_num = XY0[-u]*(2**p)
    Y0_num_Sum += Y0_num
    u = u+1
    p = p+1


Decoded_X = (X0_num_Sum * Precision_X) + a_X
Decoded_Y = (Y0_num_Sum * Precision_Y) + a_Y

print()
print(Decoded_X)
print(Decoded_Y)

# FUNCION OBJETIVO
#Calculamos la función objetivo tras decodificar X e Y.
OF_So_Far = 0.5*((Decoded_X**4)+(Decoded_Y**4)-(16*Decoded_X**2)-
                 (16*Decoded_Y**2)+(5*Decoded_X)+(Decoded_Y*5))

print("OF_So_Far:",OF_So_Far)

Counter_int = 0

for i in range(pop):
    X0_num_Sum_int = 0
    Y0_num_Sum_int = 0
        
    C_int = n_list[Counter_int]
    
    z = 0
    t = 1

    for i in range(len(XY0)//2):
        X0_num_int = C_int[-t]*(2**z)
        X0_num_Sum_int += X0_num_int
        t = t+1
        z = z+1
        
    p = 0
    u = 1 + (len(XY0)//2)
    
    for j in range(len(XY0)//2):
        Y0_num_int = C_int[-u]*(2**p)
        Y0_num_Sum_int += Y0_num_int
        u = u+1
        p = p+1
        

    Decoded_X_int = (X0_num_Sum_int * Precision_X) + a_X
    Decoded_Y_int = (Y0_num_Sum_int * Precision_Y) + a_Y
    
    OF_So_Far_int =  0.5*((Decoded_X_int**4)+(Decoded_Y_int**4)-(16*Decoded_X_int**2)-
                 (16*Decoded_Y_int**2)+(5*Decoded_X_int)+(Decoded_Y_int*5))
    '''
    print("OF_So_Far_int:",OF_So_Far_int)
    '''  
    Counter_int = Counter_int+1


Final_Best_in_Generation_X = []
Worst_Best_in_Generation_X = []

For_Plotting_the_Best = np.empty((0,len(XY0)+1))

One_Final_Guy = np.empty((0,len(XY0)+2))
One_Final_Guy_Final = []

Min_for_all_Generations_for_Mut_1 = np.empty((0,len(XY0)+1))
Min_for_all_Generations_for_Mut_2 = np.empty((0,len(XY0)+1))

Min_for_all_Generations_for_Mut_1_1 = np.empty((0,len(XY0)+2))
Min_for_all_Generations_for_Mut_2_2 = np.empty((0,len(XY0)+2))

Min_for_all_Generations_for_Mut_1_1_1 = np.empty((0,len(XY0)+2))
Min_for_all_Generations_for_Mut_2_2_2 = np.empty((0,len(XY0)+2))

n_fit_val = np.empty((0,len(XY0)))
n_fit_val_index = np.empty((0,1))


Generation = 1 

# FUNCION DE PENALIZACION
def Penalty(Array,Penalty_Value):
    Add_Penalties = []
    a_X1 = -5 # Lower bound of X
    b_X1 = 5 # Upper bound of X
    l_X1 = (len(Array)//2) # Length of Chrom. X
    a_Y1 = -5 # Lower bound of Y
    b_Y1 = 5 # Upper bound of Y
    l_Y1 = (len(Array)//2) # Length of Chrom. Y
    Precision_X1 = (b_X1 - a_X1)/((2**l_X1)-1)
    Precision_Y1 = (b_Y1 - a_Y1)/((2**l_Y1)-1)
    z = 0
    t = 1
    X0_num_Sum1 = 0
    for i in range(len(Array)//2):
        X0_num1 = Array[-t]*(2**z)
        X0_num_Sum1 += X0_num1
        t = t+1
        z = z+1
    p = 0
    u = 1 + (len(Array)//2)
    Y0_num_Sum1 = 0
    for j in range(len(Array)//2):
        Y0_num1 = Array[-u]*(2**p)
        Y0_num_Sum1 += Y0_num1
        u = u+1
        p = p+1
    Decoded_X1 = (X0_num_Sum1 * Precision_X1) + a_X1
    Decoded_Y1 = (Y0_num_Sum1 * Precision_Y1) + a_Y1
    
    if Decoded_X1 > 5 or Decoded_X1 < -5:
        Pen = Penalty_Value
        Add_Penalties = np.append(Add_Penalties,Pen)
    if Decoded_Y1 > 5 or Decoded_Y1 < -5:
        Pen = Penalty_Value
        Add_Penalties = np.append(Add_Penalties,Pen)
    Sum_Add_Penalties = sum(Add_Penalties)
    return Sum_Add_Penalties
        
def Decode(Array):
        a_X2 = -5 # Lower bound of X
        b_X2 = 5 # Upper bound of X
        l_X2 = (len(Array)//2) # Length of Chrom. X
        a_Y2 = -5 # Lower bound of Y
        b_Y2 = 5 # Upper bound of Y
        l_Y2 = (len(Array)//2) # Length of Chrom. Y
        Precision_X2 = (b_X2 - a_X2)/((2**l_X2)-1)
        Precision_Y2 = (b_Y2 - a_Y2)/((2**l_Y2)-1)
        z = 0
        t = 1
        X0_num_Sum2 = 0
        for i in range(len(Array)//2):
            X0_num2 = Array[-t]*(2**z)
            X0_num_Sum2 += X0_num2
            t = t+1
            z = z+1
        p = 0
        u = 1 + (len(Array)//2)
        Y0_num_Sum2 = 0
        for j in range(len(Array)//2):
            Y0_num2 = Array[-u]*(2**p)
            Y0_num_Sum2 += Y0_num2
            u = u+1
            p = p+1
        Decoded_X2 = (X0_num_Sum2 * Precision_X2) + a_X2
        Decoded_Y2 = (Y0_num_Sum2 * Precision_Y2) + a_Y2
        OF_So_Far2 = 0.5*((Decoded_X2**4)+(Decoded_Y2**4)-(16*Decoded_X2**2)-
                         (16*Decoded_Y2**2)+(5*Decoded_X2)+(Decoded_Y2*5))
        return Decoded_X2,Decoded_Y2,OF_So_Far2

Best_Guy_So_Far = []

for i in range(gen):
    
    Counter_1 = 0
    
    New_Population = np.empty((0,len(XY0))) # Saving the new generation
    
    All_in_Generation_X_1 = np.empty((0,len(XY0)+1))
    All_in_Generation_X_2 = np.empty((0,len(XY0)+1))
    
    Min_in_Generation_X_1 = []
    Min_in_Generation_X_2 = []
    
    Save_Best_in_Generation_X = np.empty((0,len(XY0)+1))
    Final_Best_in_Generation_X = []
    Worst_Best_in_Generation_X = []
    
    n_fit_val_pop = np.empty((0,len(XY0)))
    n_fit_val_index_pop = np.empty((0,1))
    
    
    print()
    print("--> GENERATION: #",Generation)
    
    Family = 1

    for j in range(int(pop/2)): # range(int(pop/2))
            
        print()
        print("--> FAMILY: #",Family)
              
            
        # Tournament Selection
        # Tournament Selection
        # Tournament Selection
        
        Parents = np.empty((0,len(XY0)))
        
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
    
            Decoded_X_W1 = Decode(Warrior_1)[0]
            Decoded_Y_W1 = Decode(Warrior_1)[1]
            OF_So_Far_W1 = Decode(Warrior_1)[2]
            
            Prize_Warrior_1 = OF_So_Far_W1
            
                     
            # For Warrior #2
    
            Decoded_X_W2 = Decode(Warrior_2)[0]
            Decoded_Y_W2 = Decode(Warrior_2)[1]
            OF_So_Far_W2 = Decode(Warrior_2)[2]
            
            Prize_Warrior_2 = OF_So_Far_W2
      
    
            # For Warrior #3
    
            Decoded_X_W3 = Decode(Warrior_3)[0]
            Decoded_Y_W3 = Decode(Warrior_3)[1]
            OF_So_Far_W3 = Decode(Warrior_3)[2]
            
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
        
        # Pick parents based on cumalitive probablities   
        
        Parent_1 = Parents[0]
        Parent_2 = Parents[1]
        
        
        # Crossover
        # Crossover
        # Crossover
        
        
        Child_1 = np.empty((0,len(XY0)))
        Child_2 = np.empty((0,len(XY0)))
        
        
        # Where to crossover
        # Two-point crossover
        
        Ran_CO_1 = np.random.rand()
        
        if Ran_CO_1 < p_c:
        
            Cr_1 = np.random.randint(0,len(XY0))
            Cr_2 = np.random.randint(0,len(XY0))
                
            while Cr_1 == Cr_2:
                Cr_2 = np.random.randint(0,len(XY0))
            
            if Cr_1 < Cr_2:
                
                Cr_2 = Cr_2 + 1
                
                Copy_1 = Parent_1[:]
                Mid_Seg_1 = Parent_1[Cr_1:Cr_2]
                
                Copy_2 = Parent_2[:]
                Mid_Seg_2 = Parent_2[Cr_1:Cr_2]
                
                First_Seg_1 = Parent_1[:Cr_1]
                Second_Seg_1 = Parent_1[Cr_2:]
                
                First_Seg_2 = Parent_2[:Cr_1]
                Second_Seg_2 = Parent_2[Cr_2:]
                
                Child_1 = np.concatenate((First_Seg_1,Mid_Seg_2,Second_Seg_1))
                Child_2 = np.concatenate((First_Seg_2,Mid_Seg_1,Second_Seg_2))
            
            else:
                
                Cr_1 = Cr_1 + 1
                
                Copy_1 = Parent_1[:]
                Mid_Seg_1 = Parent_1[Cr_2:Cr_1]
                
                Copy_2 = Parent_2[:]
                Mid_Seg_2 = Parent_2[Cr_2:Cr_1]
                
                First_Seg_1 = Parent_1[:Cr_2]
                Second_Seg_1 = Parent_1[Cr_1:]
                
                First_Seg_2 = Parent_2[:Cr_2]
                Second_Seg_2 = Parent_2[Cr_1:]
                
                Child_1 = np.concatenate((First_Seg_1,Mid_Seg_2,Second_Seg_1))
                Child_2 = np.concatenate((First_Seg_2,Mid_Seg_1,Second_Seg_2))
        
        else:
            
            Child_1 = Parent_1
            Child_2 = Parent_2
            
       
        
        # Mutation Child #1
        # Mutation Child #1
        # Mutation Child #1
        
        Mutated_Child_1 = []
        
        t = 0
        
        for i in Child_1:
        
            Ran_Mut_1 = np.random.rand() # Probablity to Mutate
            
            if Ran_Mut_1 < p_m: # If probablity to mutate is less than p_m, then mutate
                
                if Child_1[t] == 0:
                    Child_1[t] = 1
                else:
                    Child_1[t] = 0
                t = t+1
            
                Mutated_Child_1 = Child_1
                
            else:
                Mutated_Child_1 = Child_1
        
        
        # Mutation Child #2
        # Mutation Child #2
        # Mutation Child #2
        
        Mutated_Child_2 = []
        
        t = 0
        
        for i in Child_2:
        
            Ran_Mut_2 = np.random.rand() # Probablity to Mutate
            
            if Ran_Mut_2 < p_m: # If probablity to mutate is less than p_m, then mutate
                
                if Child_2[t] == 0:
                    Child_2[t] = 1
                else:
                    Child_2[t] = 0
                t = t+1
            
                Mutated_Child_2 = Child_2
                
            else:
                Mutated_Child_2 = Child_2
        '''
        print()
        print("Mutated_Child #2:",Mutated_Child_2)
        '''
        
        # Calculate fitness values of mutated children
        
        fit_val_muted_children = np.empty((0,2))
        
        
        # For mutated child #1
        
        OF_So_Far_MC_1 = Decode(Mutated_Child_1)[2]
            
    
        # For mutated child #2
    
        OF_So_Far_MC_2 = Decode(Mutated_Child_2)[2]
        
        
        print()
        print("Before Penalty FV at Mutated Child #1 at Gen #",Generation,":", OF_So_Far_MC_1)
        print("Before Penalty FV at Mutated Child #2 at Gen #",Generation,":", OF_So_Far_MC_2)
        
            
            
        P1 = Penalty(Mutated_Child_1,20)
        OF_So_Far_MC_1 = OF_So_Far_MC_1 + P1
        
        P2 = Penalty(Mutated_Child_2,20)
        OF_So_Far_MC_2 = OF_So_Far_MC_2 + P2
            
            
                    
        print()
        print("After Penalty FV at Mutated Child #1 at Gen #",Generation,":", OF_So_Far_MC_1)
        print("After Penalty FV at Mutated Child #2 at Gen #",Generation,":", OF_So_Far_MC_2)
              
        All_in_Generation_X_1_1_temp = Mutated_Child_1[np.newaxis]
        All_in_Generation_X_1_1 = np.column_stack((OF_So_Far_MC_1, All_in_Generation_X_1_1_temp))
        
        All_in_Generation_X_2_1_temp = Mutated_Child_2[np.newaxis]
        All_in_Generation_X_2_1 = np.column_stack((OF_So_Far_MC_2, All_in_Generation_X_2_1_temp))
        
        All_in_Generation_X_1 = np.vstack((All_in_Generation_X_1,All_in_Generation_X_1_1))
        All_in_Generation_X_2 = np.vstack((All_in_Generation_X_2,All_in_Generation_X_2_1))
        
        
        Save_Best_in_Generation_X = np.vstack((All_in_Generation_X_1,All_in_Generation_X_2))
        
        
        New_Population = np.vstack((New_Population,Mutated_Child_1,Mutated_Child_2))
        
        t = 0
        
        R_1 = []
        for i in All_in_Generation_X_1:
            
            if (All_in_Generation_X_1[t,:1]) <= min(All_in_Generation_X_1[:,:1]):
                R1_1 = []
                R1_1 = [All_in_Generation_X_1[t,:1]]
                R_1 = All_in_Generation_X_1[t,:]
            t = t+1
            
                
        A_1 = min(All_in_Generation_X_1[:,:1])
        
        Min_in_Generation_X_1 = R_1[np.newaxis]
        
        t = 0
        R_2 = []
        for i in All_in_Generation_X_2:
            
            if (All_in_Generation_X_2[t,:1]) <= min(All_in_Generation_X_2[:,:1]):
                R1_2 = []
                R1_2 = [All_in_Generation_X_2[t,:1]]
                R_2 = All_in_Generation_X_2[t,:]
            t = t+1
                
        A_2 = min(All_in_Generation_X_2[:,:1])
        
        Min_in_Generation_X_2 = R_2[np.newaxis]
        
        
        Family = Family+1
    
    t = 0
    R_Final = []
    
    for i in Save_Best_in_Generation_X:
        
        if (Save_Best_in_Generation_X[t,:1]) <= min(Save_Best_in_Generation_X[:,:1]):
            R1_2_Final = []
            R1_2_Final = [Save_Best_in_Generation_X[t,:1]]
            R_Final = Save_Best_in_Generation_X[t,:]
        t = t+1
            
    A_2_Final = min(Save_Best_in_Generation_X[:,:1])
    
    Final_Best_in_Generation_X = R_Final[np.newaxis]
    '''
    print("Final_Best_in_Generation_X:",Final_Best_in_Generation_X)
    '''
    For_Plotting_the_Best = np.vstack((For_Plotting_the_Best,Final_Best_in_Generation_X))
    
    t = 0
    R_22_Final = []
    
    for i in Save_Best_in_Generation_X:
        
        if (Save_Best_in_Generation_X[t,:1]) >= max(Save_Best_in_Generation_X[:,:1]):
            R1_22_Final = []
            R1_22_Final = [Save_Best_in_Generation_X[t,:1]]
            R_22_Final = Save_Best_in_Generation_X[t,:]
        t = t+1
            
    A_22_Final = max(Save_Best_in_Generation_X[:,:1])
    
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
    
    n_fit_val = np.empty((0,len(XY0)))
    n_fit_val_index = np.empty((0,1))

    
    Best_Guy_So_Far = np.vstack((Min_for_all_Generations_for_Mut_1_1_1,Min_for_all_Generations_for_Mut_2_2_2))
    
       
    
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

print("Min in all Generations:",One_Final_Guy_Final.tolist())

print("The Lowest Cost is:",One_Final_Guy_Final[:,1])


Look = (One_Final_Guy_Final[:,1])

plt.plot(For_Plotting_the_Best[:,0])
plt.axhline(y=Look,color="r",linestyle='--')
plt.title("Z Reached Through Generations",fontsize=20,fontweight='bold')
plt.xlabel("Generations",fontsize=18,fontweight='bold')
plt.ylabel("Z",fontsize=18,fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
xyz=(Generation/4, Look)
xyzz = (Generation/3.7, Look+0.5)
plt.annotate("Minimum Reached at: %0.3f" % Look, xy=xyz, xytext=xyzz,
             arrowprops=dict(facecolor='black', shrink=1,width=1,headwidth=5),
             fontsize=12,fontweight='bold')
plt.show()


print()
print("Final Solution:",One_Final_Guy_Final[:,2:])
print("The Lowest Z is:",One_Final_Guy_Final[:,1])
print("At Generation:",One_Final_Guy_Final[:,0])

XY0_Encoded_After = Final_Here[2:]


# DECODING
# DECODING
# DECODING

Decoded_X_After = Decode(XY0_Encoded_After)[0]
Decoded_Y_After = Decode(XY0_Encoded_After)[1]

print()
print("X After:",Decoded_X_After)
print("Y After:",Decoded_Y_After)



