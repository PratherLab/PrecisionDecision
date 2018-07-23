% Percision Decision Kids (or adults) 11.23.16
% NO EVOL ALGO IMPLIMENTATION
% DEMO OF ANS/EST DATA WHEN VARYING SPECS
% NO SUBJECT FIT
%{
Model for the Percision/Decision project
Based on the adult version with slight changes due to lack
of ERP data with children. 
This model completes both a comparison and free estimation task.
Model completes both task with the same specifications
results are compared to participant data.
The set up of this model is slightly different than TBT.
There is no training + test phase. 
This is more like a cluster analysis. What clusters of specs can 
account for each Pps data on both tasks simultaneously.

Specs include, percision of tuning curves and competition dynamics
of the decision layer. 
All subs at once.
write data to seperate files as you go
(combine from the shell)
maybe make two versions, Perc & Desc.
where fit is one of two only. and compare how well
Perc and Desc models fit particpants data
but also show that Perc and Dec can both fit both tasks
equally well if they have to just do one. but only one
of Perc or Desc can do both well simultaneously. that is the winner

1) Global and Subject Parameter Setting
    -set Evolved Parameters here. 
    -set constants
 2) Simmulation: Timesteps
    -Train or Test selection (use of learning or not)
 3) Adjustments to Kernels
    -change based on trial ERP 
 4) repeat 2-3
 5) Data Processing: Visualization/Plotting

%}


%{
%The math governing the layers is taken from example files and from Simmering et al. 2008 also Samuelson
%2009
%Functions called: Gauss, GaussNorm, Sigmoid
 
% Basic Architecture Set up perceptual input fields that conver several
% perceptual features for 2 objects. Intermediate Percept field for each
% stimuli. Decision layer(U) that is competative and selects from two
% inputs.
%
% U layer self exc & Inh are set up similar to Sameulson to create highly
% competative one peak wins dynamics
%
% Examples:
% kernel_uu: interaction kernel for connection u->u (exc + inh)
% kernel_nu: interaction kernel from n to u

%FITNESS DEFINITION AT LINE 583
%}
%%%%%%%%%%%%%%%%%%%%
%1. Initial model parameters 
%%%%%%%%%%%%%%%%%%%%

clear; %Reset random seed
tic;
%Density to be added
fieldSize = 181; % must be odd

%load kidestdata.csv %load human freeest data (order, sub, trialorder, label, reponse, key, ratio
load MTestdata.csv
% difference, diff sq, diff abs, ratio error, outlier
%load kidansdata.csv  % load human comparison data 
load MTansdata.csv
%(order, sub, RT, response, Label, trialOrder, key, diff, corretness, ratio, A, B, Ratio, label   
load bestntcspec.csv 
% load pilot EEG data. coloums trial number, Ochannel, P7, P3, Averaged, A,
% B, Ratio, RT, response correctness

load estkey.csv


prompt = {'Enter number of Batches(10Subs):','Comparison trials per sub:','Estimation trials:','SubjectStart','No. of Subs'};% Dialog Box Input
dlg_title = 'Input';
num_lines = 1;
def = {'1','96','64','1','5'};
answer = inputdlg(prompt,dlg_title,num_lines,def);
N= 10; %Rounds per batch

specA = str2double(answer);

%MODE SELECTION TEST OR TRAIN

prompt = {'Train(1) or Test(2)','Percision(1) or Desicion(2)','NTC'};% Dialog Box Input
dlg_title = 'Input';
num_lines = 1;
def = {'1','1','1'};
answer = inputdlg(prompt,dlg_title,num_lines,def);
specB = str2double(answer); %RENAME
SubTotal = specA(5,1); % total subjects to run
SubStart = specA(4,1); %subject to start with
batches = specA(1,1); %number of batches, equal to 10 subject runs.
bestruns = zeros(SubTotal,20);
totalruns = zeros(SubTotal*N*batches,20);

SubID = specA(4,1); %subject ID
ntcw = specB(3,1);
display(SubStart)
display(SubTotal)



for CurrentSub = 1:SubTotal
%put Subject LOOP HERE?
%cd('/Users/Richard/Documents/MATLAB')
time_u = 500; % time constant of dynamic field defined trial by trial
time_mu_build = 500; time_mu_decay = 2000; % time constants of memory traces USE DECAY IN DECISION FIELD
time_p1_build = 500; %build variables. start with slight bias to P1 being faster. Bigger is slower
time_p2_build = 450;
time_p1_decay = 700;
time_p2_decay = 700;

b_u = 4; % steepness parameter of sigmoid function (S curve) for..the activation calc. larger = step function, smaller = linear
h_u = -2; % resting level for decision layer, AKA height
h_p = -2; 
h_pb = -2; %resting level for perceptual layers
h_ps = -2;
n_u = 0; %.005; % noise level for U-layer
n_p = .1; %noise for other layers .005
w_n = .5; % width of the noise kernel.
erpfactor = 1;


display(CurrentSub)
nTrials = specA(2,1).*batches; %nunber of total trials
FreeTrials = specA(3,1).*batches;
subtrials = specA(2,1); % trials per subject ANS
subtrialsEST = specA(3,1); %trials per subject EST
tMax = 500; % Time steps per trial
totaltime = subtrials*tMax; % timesteps per subject ANS
totaltimeEST = subtrialsEST*tMax; %timesteps per sub EST
tStoreFields = 1:totaltime; % use this to store field activities for each subject
tStoreFieldsEST = 1:totaltimeEST;
tStimulusStart = 20; %start and End of stim presentation per trial
tStimulusEnd = 450;  %may need an inter trial inverval



%set current data to the Subject

tbthumandata = zeros(96,15);
esthumandata = zeros(64,13);
c = 1;
d = 1;


    for a = 1:length(MTansdata) 
        if MTansdata(a,15) == SubID %MAY NEED TO CHANGE SUB IDS TO
                                    %NUMBER 1 - 30 WITH NO
                                    %SKIP. ADD COLUMN TO SUM DATA FILES
            tbthumandata(c,:) = MTansdata(a,:);
            
            c=c+1;
        end
            
    end
    
    for a = 1:length(MTestdata);
    if MTestdata(a,13) ==SubID
        esthumandata(d,:) = MTestdata(a,:);
        d = d+1;
    end
    end
    
esthumandata(:,6) = estkey(1:64,1); %puts EST trials in order


% ANS DATA
modeldata = zeros(nTrials,7,10); %data for model decision & RTs and variables for each trial for 10 subjects
modelpop = zeros(10,12); %pop specs for most recent batch
totalpop = zeros(batches*10,20); %every subject for every batch specs and error
popdata = zeros(10*batches,10); %data archive for all batch specs
gengraph = zeros(batches,4); %a graph of how the min error changes across batches

% EST DATA
modeldataEST = zeros(FreeTrials,7,10); %data for model Estimations and target values each trial for 10 subjects
modelpopEST = zeros(10,12); %pop specs for most recent batch
totalpopEST = zeros(batches*10,15); %every subject for every batch specs and error
popdataEST = zeros(10*batches,10); %data archive for all batch specs
gengraphEST = zeros(batches,4); %a graph of how the min error changes across batches


%Decision Layer Parameters
% to get competative peak make inhib strong and wide? exc strong and narrow
s_exc = 80; w_exc = 3; %self exhitation U layer
s_inh = 70; w_inh = 25; g_inh = 0; %U layer inhibition, global 275 default

%Stim1 Layers Parameters: Area, Density, Number Percept
s_excAS1 = 5; w_excAS1 = 10; % Strength/Width of ex connection from A1
s_excNS1 = 40; w_excNS1 = 10; % Strength/Width of ex connection from N1
s_excPS1 = 25; w_excPS1 = 6; % S W of ex connection from Percept1 to U 25 default
s_excDS1 = 5; w_excDS1 = 10; %strength/widgth of ex connection from D1

s_inhAS1 = 1; w_inhAS1 = 30; % A layer inhibition to Percept1
s_inhNS1 = 1; w_inhNS1 = 30; % N layer inhibition to Percept1
s_inhDS1 = 1; w_inhDS1 = 30; % D layer inhibition to Percept1
s_inhPS1 = 35; w_inhPS1 = 25; % P layer inhibition to U 

%Stim2 Layers Parameters: Area, Number Percept
s_excAS2 = 5; w_excAS2 = 10; % Strength of ex connection from A2
s_excNS2 = 40; w_excNS2 = 10; % S W of ex connection from N
s_excPS2 = 25; w_excPS2 = 6; % S W of ex connection from Percept2 to U 40 default
s_excDS2 = 10; w_excDS2 = 30; %strength/widgth of ex connection from D1

s_inhAS2 = 1; w_inhAS2 = 30; % A layer inhibition to Percept2
s_inhNS2 = 1; w_inhNS2 = 30; % N layer inhibition to Percept2
s_inhDS2 = 1; w_inhDS2 = 30; % D layer inhibition to Percept2
s_inhPS2 = 35; w_inhPS2 = 25; % P layer inhibition to U

%%%%%%%%%%%%%%%%
Uratio = s_exc/s_inh; %ratio of P1 and P2 connection strengths to U layer only used to increase P1
width = 2;

%%%%%%%%%%%%%%%%%%%%

pop = zeros(10,12); %keeps track of variables for 10 subjects worth.
pop(1,1) = time_u; %U layer time 
pop(1,2) = s_inh; %ulayer inh strength (higher is more winner take all)
pop(1,3) = n_u; %noise in Ulayer
pop(1,4) = b_u; %steepness ouf output function. range 1.1 to 5?
pop(1,5) = time_p2_build;
pop(1,6) = s_excPS2;
pop(1,7) = w_inh; % U layer self inhibition width ?
pop(1,8) = 2; %tuning curve width factor

for w = 1:10
pop(w,1) = time_u; %randi([300,700],1,1);
pop(w,2)= randi([1,300],1,1); %s_inh
pop(w,3) = n_u; %randi([1,500])*.0005; % try .0005. range effects correlations
pop(w,4) = b_u; %randi([11,50])*.1;    
pop(w,5) =  randi([250,500]); %time_p2_build;
pop(w,6) = s_excPS2; %randi([25,120]);
pop(w,7) = randi([3,40]); %w_inh
pop(w,8) = randi([150,380])*.01; %ntc precision
end



%Percicion vs. Decision model traiing
if specB(2,1) == 1
    evoalgo = 1;
    pop(1:10,5) = time_p2_build;
    pop(1:10,2) = s_inh;
    pop(1:10,6) = s_excPS2;
    pop(1:10,7) = w_inh;
else
    evoalgo = 2;
    pop(1:10,2) = s_inh;
    pop(1:10,6) = s_excPS2;
    pop(1:10,7) = w_inh;
    ntcw = bestntcspec(CurrentSub,1);
    pop(1:10,8) = ntcw; %bestntcspec(CurrentSub,1); %set all NTC values to best prior
end

%if evoalgo = 1 then set pop(1:10,2) to default s_inh here and at the batch
%
%if evoalgo = 2 then set pop(1:10,8) to ntcw dedault here at at the turn


%Human Constants.
subRT = mean(tbthumandata(:,3)); % Mean reaction time
subSD = std(tbthumandata(:,3)); %Standard Deviation RT
wrongs = subtrials - sum(tbthumandata(1:subtrials,9)); %wrong human answers
negmatch = zeros(10,2);

ESTaccuarcy = mean(esthumandata(:,10));%subjects mean absolute deviation from target value


% INSTANIATION LOOP

for b = 1:batches

disp(b)    
deviations = zeros(subtrials,7, 10); %deviation between model and human data ANS
deviationsEST = zeros(subtrialsEST, 7, 10); % same for EST
    

%%%%%%%%%%%%%%%%%%
%2  INITIALIZATION OF LAYERS %
%%%%%%%%%%%%%%%%%%

%Data Arrays layer activations 
data_u = zeros( length(tStoreFields), fieldSize, N); % Decision layer data
data_P1 = zeros( length(tStoreFields), fieldSize, N); % Percept 1 data
data_P2 = zeros( length(tStoreFields), fieldSize, N); % Percept 2 data
data_N1 = zeros( length(tStoreFields), fieldSize, N); % Number 1 data
data_N2 = zeros( length(tStoreFields), fieldSize, N); % Number 2 data
data_P1EST = zeros( length(tStoreFieldsEST), fieldSize, N);
halfField = floor(fieldSize/2); % set to 90

% create row vectors for "current" field activities
field_u = zeros(1, fieldSize); %Decision layer
field_P1 = zeros(1, fieldSize); % Percept 1 layer
field_P2 = zeros(1, fieldSize); % Percept 2 layer

field_D1 = zeros(1, fieldSize); % Density 1layer
field_D2 = zeros(1, fieldSize); % Density 2 layer

field_A2 = zeros(1, fieldSize); % Area 2 layer
field_N2 = zeros(1, fieldSize); % Number 2 layer
field_A1 = zeros(1, fieldSize); % Area 1 layer
field_N1 = zeros(1, fieldSize); % Number 1 layer

actsum = zeros(totaltime,subtrials,N); %Descition layer activation data online-summary
actsump1 = zeros(totaltime,subtrials,N); %Percept layer 1 activation data online-summary
actsump2 = zeros(totaltime,subtrials,N); %Percept layer 2 activation data online-summary

actsumEST = zeros(totaltime,subtrials,N); %Descition layer activation data online-summary
actsump1EST = zeros(totaltime,subtrials,N); %Percept layer 1 activation data online-summary
actsump2EST = zeros(totaltime,subtrials,N); %Percept layer 2 activation data online-summary


%BATCH LOOP 
%%%%
%BEGIN SUBJECT LOOP FOR N=10 SUBJECTS
%%%%

for N = 1:10    % subject within a batch (b)

% set specs    
time_u = pop(N,1);
s_inh =  pop(N,2);
n_u = pop(N,3);
b_u = pop(N,4);
%time_p2_build = pop(N,5);
erpfactor = pop(N,6);
%MORE HERE FOR EST TRIAL? NTCW?
%time_p2_decay = pop(N,6);

    
    
    
% create matrices to store field activities at different times using field
% size number of time points and trials only for the current subject

history_p1 = zeros(length(tStoreFields), fieldSize); % Stim 1 
history_p2 = zeros( length(tStoreFields), fieldSize);% Stim 2 Num
history_u = zeros( length(tStoreFields), fieldSize); % decision layer
history_a1 = zeros( length(tStoreFields), fieldSize); %area layer
history_a2 = zeros( length(tStoreFields), fieldSize); %area layer
history_n1 = zeros( length(tStoreFields), fieldSize); % num layer
history_n2 = zeros( length(tStoreFields), fieldSize); % num layer
history_d1 = zeros( length(tStoreFields), fieldSize); % den layer
history_d2 = zeros( length(tStoreFields), fieldSize); % den layer

%Estimation task history
history_p1EST = zeros(length(tStoreFieldsEST), fieldSize); % Stim 1 
history_p2EST = zeros( length(tStoreFieldsEST), fieldSize);% Stim 2 Num
history_uEST = zeros( length(tStoreFieldsEST), fieldSize); % decision layer
    
    currenttrial = 1; % trials counter for each subject

% index of the current position in the history matrices
iHistory = 1;

%%%%%%%%%%%%%%%%
%INTERACTION KERNEL SETUP
%%%%%%%%%%%%%

%kernel is 1xfieldzise vector make by adding together (strengh_exc *
%guassNorm(width_exc) - (strenth_inhib * guassNorm(width_inhib) minus
%global inhib

kernel_uu = s_exc * gaussNorm(-halfField:halfField, 0, w_exc) ...
  - s_inh * gaussNorm(-halfField:halfField, 0, pop(N,7)) - g_inh ;  %Decision layer self exhit



%Percept 1 Kernels

kernel_pp1 = s_exc * gaussNorm(-halfField:halfField, 0, w_exc) ...
  - s_inhPS1 * gaussNorm(-halfField:halfField, 0, w_inhPS1) - g_inh ; % Percept to Percept
kernel_pa1 = s_excAS1 * gaussNorm(-halfField:halfField, 0, w_excAS1) ...
  - s_inhAS1 * gaussNorm(-halfField:halfField, 0, w_inhAS1) - g_inh; %Area to Percept
kernel_pn1 = s_excNS1 * gaussNorm(-halfField:halfField, 0, w_excNS1) ...
  - s_inhNS1 * gaussNorm(-halfField:halfField, 0, w_inhNS1) - g_inh; %Number to Percept
kernel_pd1 = s_excDS1 * gaussNorm(-halfField:halfField, 0, w_excNS1) ...
 - s_inhDS1 * gaussNorm(-halfField:halfField, 0, w_inhDS1) - g_inh; %Density to Percept
kernel_up1 = s_excPS1 * gaussNorm(-halfField:halfField, 0, w_excPS1) ...
  - s_inhPS1 * gaussNorm(-halfField:halfField, 0, w_inhPS1) - g_inh; %Percept to Decision


%Percept 2 Kernels
kernel_pp2 = s_exc * gaussNorm(-halfField:halfField, 0, w_exc) ...
  - s_inhPS2 * gaussNorm(-halfField:halfField, 0, w_inhPS1) - g_inh ; %what Inh to use??
kernel_pa2 = s_excAS2 * gaussNorm(-halfField:halfField, 0, w_excAS2) ...
  - s_inhAS2 * gaussNorm(-halfField:halfField, 0, w_inhAS2) - g_inh; %Area to Percept
kernel_pn2 = s_excNS2 * gaussNorm(-halfField:halfField, 0, w_excNS2) ...
  - s_inhNS2 * gaussNorm(-halfField:halfField, 0, w_inhNS2) - g_inh; %Number to Percept
kernel_pd2 = s_excDS2 * gaussNorm(-halfField:halfField, 0, w_excNS2) ...
 - s_inhDS2 * gaussNorm(-halfField:halfField, 0, w_inhDS2) - g_inh; %Density to Percept
kernel_up2 = pop(N,6) * gaussNorm(-halfField:halfField, 0, w_excPS2) ...
  - s_inhPS2 * gaussNorm(-halfField:halfField, 0, w_inhPS2) - g_inh; %Percept to Decision

% set up the kernel for correlated noise (if required)
%this makes noise centralized on a gaussian??
if w_n > 0
  kernel_q = gaussNorm(-halfField:halfField, 0, w_n);
end  



 

field_u(1:fieldSize) = h_u;
field_P1(1:fieldSize) = h_pb;
field_P2(1:fieldSize) = h_ps;
  
field_N1(1:fieldSize) = h_p;
field_A1(1:fieldSize) = h_p;
field_D1(1:fieldSize) = h_p;
  
field_N2(1:fieldSize) = h_p;
field_A2(1:fieldSize) = h_p;
field_D2(1:fieldSize) = h_p;

  
  %%%%%%%%%% 
  % 4. 1 One Subject
  %%%%%%%%
  
  %%%%%%%%
  % Comp Task
  
  
for i = 1 : (subtrials)

 % TRIAL BY TRIAL ERP BASED CONNECTION/KERNEL ADJUSTMENTS
%Adjustment of Percept1 to U layer connection based on current trial ERP
%measure. If the ERP is off from expected (positive value) then strengen
%the connection of P1 to U later which inc liklihood for incorrect answer
% the amount of the change is an Evolved Parameter.
%s_excPS1 = s_excPS2;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TRIAL BY TRIAL ERP ADJUSTMENTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%{
% EXPECTED ERP MEASURE
if ERPdev(i,2) > 10     
    s_excPS1 = s_excPS1 * (ERPdev(i,2)+1);   
end

% RAW ERP MEASURE
if tbthumandata(i,9) < 1.7 && tbthumandata(i,2) ~= 0
    s_excPS1 = s_excPS1 - ((tbthumandata(i,2) - smallERP)* pop(N,6)) ; 
end


  kernel_up1 = s_excPS1 * gaussNorm(-halfField:halfField, 0, w_excPS1) ...
  - s_inhPS1 * gaussNorm(-halfField:halfField, 0, w_inhPS1) - g_inh; %Percept to Decision

    
    
  %}  
    
trialnum = i;
% area and number stim
a1 = tbthumandata(mod(trialnum-1,subtrials)+1,11);
a2 = tbthumandata(mod(trialnum-1,subtrials)+1,2);
a3 = tbthumandata(mod(trialnum-1,subtrials)+1,12);
a4 = tbthumandata (mod(trialnum-1,subtrials)+1,4);
n1 = tbthumandata(mod(trialnum-1,subtrials)+1,11); %STIM 1
n2 = (pop(N,8))*((tbthumandata (mod(trialnum-1,subtrials)+1, 11))/25 + 5) ; %Width this needs to be adjustable
n3 = tbthumandata(mod(trialnum-1,subtrials)+1,12); %STIM 2
n4 = (pop(N,8))*((tbthumandata(mod(trialnum-1,subtrials)+1, 12))/25 + 5); %Width


%n2 = n2* 2./ (1 + exp(-.2*(tbthumandata(mod(trialnum-1,subtrials)+1,5)))); %sigmoid adjustment where mean = 1 assymtops on 0 and 2.
%n2 = 2;
%n4 = n2;

stimN1 = (181-n1)*.1*gauss(1:fieldSize, n1, n2); % a localized input (location, size)
stimN2 = (181-n3)*.1*gauss(1:fieldSize, n3, n4); % a broad localized input
stimA1 = 6*gauss(1:fieldSize, a1, a2); % a localized input
stimA2 = 6*gauss(1:fieldSize, a3, a4); % a localized input

 


  % prepare matrix that holds the stimulus for each time step
  stimulusN1 = zeros(totaltime, fieldSize);
  stimulusN2 = zeros(totaltime, fieldSize);
  stimulusA1 = zeros(totaltime, fieldSize);
  stimulusA2 = zeros(totaltime, fieldSize);
  stimulusD1 = zeros(totaltime, fieldSize);
  stimulusD2 = zeros(totaltime, fieldSize);
  
  % if needed, create a new stimulus pattern for every trial
  %not using this?
  stimPos = round(fieldSize * rand);
  stimPos2 = round(fieldSize * rand);
  stimT = 6*gauss(1:fieldSize, stimPos, 5);
  stimT2 = 6*gauss(1:fieldSize, stimPos2, 5);
  
  % write the stimulus pattern into the stimulus matrix for all time steps
  % where it should be active
  for j = tStimulusStart : tStimulusEnd
    stimulusN1(j, :) = stimN1;
    stimulusN2(j, :)= stimN2;
    stimulusA1(j, :) = stimA1;
    stimulusA2(j, :) = stimA2;
    
  end
  
  
  %%%%%%%%%%%%%%%%%%%%%%
  %  4.1  1 TRIAL : TIMESTEP LOOP
  %%%%%%%%%%%%%%%%%%%
  
  
  for t = 1 : tMax
    % calculation of field outputs
    output_u = sigmoid(field_u, b_u, 0);
    output_p1 = sigmoid(field_P1, b_u, 0);
    output_p2 = sigmoid(field_P2, b_u, 0);
    output_a1 = sigmoid(field_A1, b_u, 0);
    output_n1 = sigmoid(field_N1, b_u, 0);
    output_a2 = sigmoid(field_A2, b_u, 0);
    output_n2 = sigmoid(field_N2, b_u, 0);
    %output_d1 = sigmoid(field_D1, b_u, 0);
    %output_d2 = sigmoid(field_D2, b_u, 0);
    % circular padding of outputs for convolution
    % just creates a repeat string ouf output?
    output_u_padded = [output_u(halfField+2:fieldSize), output_u, output_u(:, 1:halfField)];
    output_p1_padded = [output_p1(halfField+2:fieldSize), output_p1, output_p1(:, 1:halfField)];
    output_p2_padded = [output_p2(halfField+2:fieldSize), output_p2, output_p2(:, 1:halfField)];
    output_a1_padded = [output_a1(halfField+2:fieldSize), output_a1, output_a1(:, 1:halfField)];
    output_n1_padded = [output_n1(halfField+2:fieldSize), output_n1, output_n1(:, 1:halfField)];
    output_a2_padded = [output_a2(halfField+2:fieldSize), output_a2, output_a2(:, 1:halfField)];
    output_n2_padded = [output_n2(halfField+2:fieldSize), output_n2, output_n2(:, 1:halfField)];
    %output_d1_padded = [output_d1(halfField+2:fieldSize), output_d1, output_d1(:, 1:halfField)];
    %output_d2_padded = [output_d2(halfField+2:fieldSize), output_d2, output_d2(:, 1:halfField)];

    
    % get endogenous input to fields by convolving outputs with interaction kernels
    conv_uu = conv2(1, kernel_uu, output_u_padded, 'valid'); %U layer self connections
    conv_pp1 = conv2(1, kernel_pp1, output_p1_padded, 'valid'); %p1 layer self connections
    conv_pp2 = conv2(1, kernel_pp2, output_p2_padded, 'valid'); %p2 layer self connections
    conv_up1 = conv2(1, kernel_up1, output_p1_padded, 'valid');% P1 to U layer
    conv_up2 = conv2(1, kernel_up2, output_p2_padded, 'valid');% P2 to U layer
    conv_pa1 = conv2(1, kernel_pa1, output_a1_padded, 'valid');% A1 to P1
    conv_pn1 = conv2(1, kernel_pn1, output_n1_padded, 'valid');% N1 to P1
    conv_pa2 = conv2(1, kernel_pa2, output_a2_padded, 'valid'); %A2 to P2
    conv_pn2 = conv2(1, kernel_pn2, output_n2_padded, 'valid'); %N2 to P2
    %conv_pd1 = conv2(1, kernel_pd1, output_d1_padded, 'valid'); %D1 to P1
    %conv_pd2 = conv2(1, kernel_pd2, output_d2_padded, 'valid'); %D2 to P2

    
    %Perceptual Conv normalization. adjust conv such the max is the same
    cp1max = max(conv_up1); %not sure this is needed 9.21.15
    cp2max = max(conv_up2);
    cpadj = cp1max/cp2max;
    %conv_up2 = conv_up2.*cpadj*2.8;
    
    
    % create field noise for this timestep
    noise_u = n_u * randn(1, fieldSize);
    noise_p = n_p * randn(1, fieldSize);
    
    if w_n > 0 % create spatially correlated noise by convolution IS THIS USED?
      noise_u_padded = [noise_u(halfField+2:fieldSize), noise_u, noise_u(:, 1:halfField)];
      noise_u = conv2(1, kernel_q, noise_u_padded, 'valid');
      noise_p_padded = [noise_p(halfField+2:fieldSize), noise_p, noise_p(:, 1:halfField)];
      noise_p = conv2(1, kernel_q, noise_p_padded, 'valid');
    end
    
    % update field activities
    field_u = field_u + 1/time_u * (-field_u + h_u + conv_up1 + conv_up2 + conv_uu) + noise_u; % use +conv_uu ??
    field_P1 = field_P1 + 1/time_p1_build * (-field_P1 + h_pb +stimulusN1(t, :)) + noise_p; %+ conv_pp1
    field_P2 = field_P2 + 1/pop(N,5) * (-field_P2 + h_ps + stimulusN2(t, :)) + noise_p; %+ conv_pp2
    field_A1 = field_A1 + 1/time_u * (-field_A1 + h_p + stimulusA1(t, :)) + noise_p;
    field_N1 = field_N1 + 1/time_u * (-field_N1 + h_p + stimulusN1(t, :)) + noise_p;
    field_A2 = field_A2 + 1/time_u * (-field_A2 + h_p + stimulusA2(t, :)) + noise_p;
    field_N2 = field_N2 + 1/time_u * (-field_N2 + h_p + stimulusN2(t, :)) + noise_p;
    %field_D1 = field_D1 + 1/time_u * (-field_D1 + h_p + stimulusD1(t, :)) + noise_p;
    %field_D2 = field_D2 + 1/time_u * (-field_D2 + h_p + stimulusD2(t, :)) + noise_p;
    
   
    % store field activities at the selected time steps
    if any(tStoreFields == t)
      
      %history_n1(iHistory, :) = stimulusN1(t, :);
      %history_n2(iHistory, :) = stimulusN2(t,:);
      history_n2(iHistory, :) = field_N2;
      history_a2(iHistory, :) = field_A2;
      history_u(iHistory, :) = field_u;
      history_p1(iHistory, :) = field_P1;
      history_p2(iHistory, :) = field_P2;
      history_a1(iHistory, :) = field_A1;
      history_n1(iHistory, :) = field_N1;
      %history_d1(iHistory, :) = field_D1;
      %history_d2(iHistory, :) = field_D2;
      
      iHistory = iHistory + 1;
    end
    
    
     
        if 0
        indexP1 = find(field_P1(1,:)==max(field_P1(1,:))); %finds location of peak
        indexP2 = find(field_P2(1,:)==max(field_P2(1,:))); %finds location of peak
        peakP1 = max(field_P1(1,:));
        peakP2 = max(field_P2(1,:));
     
        %s_excPS1 = P1con * (1+ 1*indexP1/181);
        %s_excPS2 = P2con * (indexP2/indexP1); %the bigger value?
        end;
    
        
  end
  %END OF TIME STEP LOOP

  % clear history_u, p1 and p2 of "all infinities"
  %{
  for c = 1:tMax; 
         actsum(c,i,N) = find(history_u((c+500*(i-1)),:)==max(history_u((c+500*(i-1)),:)));
        actsump1(c,i,N) = find(history_p1((c+500*(i-1)),:)==max(history_p1((c+500*(i-1)),:)));
        actsump2(c,i,N) = find(history_p2((c+500*(i-1)),:)==max(history_p2((c+500*(i-1)),:)));
        end;  
  %}
  
  for c = 1:tMax;
      [MU,Q] = max(history_u((c+500*(i-1)),:)); %if all Inf it just reports index of first
      [MP,R] = max(history_p1((c+500*(i-1)),:));
      [MP2,S] = max(history_p2((c+500*(i-1)),:));
      actsum(c,i,N) = Q;
      actsump1(c,i,N) = R;
      actsump2(c,i,N) = S;
      
  end

  currenttrial = currenttrial+1; %trial counter


%Evaluate the decision & perceptual layer output by finding max value
% for layer at each timepoint for all trials and all subjects
%gets the index of the max activity across all timesteps for U and P1
    %and P2 layers
          
  
%Determining the Desision from the U layer        
uwindow = zeros(30,1); %defines window of recent 10 time steps for U layer
 
 for a = (tStimulusStart+50):500
 uwindow(1:30,1) = actsum(a-30:a-1,i,N); %if current values deviation from each of the last 10 is less than 3 then set.

for d = 1:30
 uwindow(d,1) = abs(uwindow(d,1) - actsum(a,i,N));
 end
 
 if max(uwindow(1:30,:)) < 3
     utime = a;
     udescision = actsum(a,i,N);
     break
 end
 end
 
% compariing model response to human responses
 %for r = 1:subtrials
     
 if abs(tbthumandata(i,11)-udescision) < abs(tbthumandata(i,12)-udescision)  %1 is smaller on every trial
     modeldata(i+(subtrials*(b-1)),1,N) = 0; %incorrect Which is Bigger?
 else
     modeldata(i+(subtrials*(b-1)),1,N) = 1; %correct
 end

modeldata(i+(subtrials*(b-1)),2,N) = utime/300 + subRT/2 +.25; %converts model time to Millisecs
modeldata(i+(subtrials*(b-1)),3,N) = time_u;
modeldata(i+(subtrials*(b-1)),4,N) = b;
modeldata(i+(subtrials*(b-1)),5,N) = udescision;
modeldata(i+(subtrials*(b-1)),6,N) = tbthumandata(i,11);
modeldata(i+(subtrials*(b-1)),7,N) = tbthumandata(i,12);

 
 field_u(1:fieldSize) = h_u;
 field_P1(1:fieldSize) = h_pb;
 field_P2(1:fieldSize) = h_ps;
  
 field_N1(1:fieldSize) = h_p;
 field_A1(1:fieldSize) = h_p;
 %field_D1(1:fieldSize) = h_p;
  
 field_N2(1:fieldSize) = h_p;
 field_A2(1:fieldSize) = h_p;
 %field_D2(1:fieldSize) = h_p;


end %END TRIALS LOOP


%%%%%%%%%%%%%%%
% FREE ESTIMATION TASK LOOP
% with the same neural fields complete the free estimation task.
% what 
iHistory = 1;
for i = 1:subtrialsEST; %number of free est trials total
%s_excPS1 = s_excPS2;
 trialnum = i;

n1 = esthumandata(mod(trialnum-1,subtrialsEST)+1,6); %STIM 1
n2 = (pop(N,8))*((esthumandata (mod(trialnum-1,subtrialsEST)+1, 6))/25 + 5) ; %Width this needs to be adjustable
stimN1 = (181-n1)*.1*gauss(1:fieldSize, n1, n2); % a localized input (location, size)

% prepare matrix that holds the stimulus for each time step
stimulusN1 = zeros(totaltime, fieldSize);

% write the stimulus pattern into the stimulus matrix for all time steps
  % where it should be active
for j = tStimulusStart : tStimulusEnd
    stimulusN1(j, :) = stimN1;
end


  %%%%%%%%%%%%%%%%%%%%%%
  % ESTIMATION 1 TRIAL : TIMESTEP LOOP
  %%%%%%%%%%%%%%%%%%%

for t = 1 : tMax
    % calculation of field outputs
    output_u = sigmoid(field_u, b_u, 0);
    output_p1 = sigmoid(field_P1, b_u, 0);
    output_p2 = sigmoid(field_P2, b_u, 0);
    
    % circular padding of outputs for convolution
    % just creates a repeat string ouf output?
    output_u_padded = [output_u(halfField+2:fieldSize), output_u, output_u(:, 1:halfField)];
    output_p1_padded = [output_p1(halfField+2:fieldSize), output_p1, output_p1(:, 1:halfField)];
    output_p2_padded = [output_p2(halfField+2:fieldSize), output_p2, output_p2(:, 1:halfField)];
    

    
    % get endogenous input to fields by convolving outputs with interaction kernels
    conv_uu = conv2(1, kernel_uu, output_u_padded, 'valid'); %U layer self connections
    conv_pp1 = conv2(1, kernel_pp1, output_p1_padded, 'valid'); %p1 layer self connections
    conv_pp2 = conv2(1, kernel_pp2, output_p2_padded, 'valid'); %p2 layer self connections
    conv_up1 = conv2(1, kernel_up1, output_p1_padded, 'valid');% P1 to U layer
    conv_up2 = conv2(1, kernel_up2, output_p2_padded, 'valid');% P2 to U layer
    
    
    %Perceptual Conv normalization. adjust conv such the max is the same
    cp1max = max(conv_up1); %not sure this is needed 9.21.15
    cp2max = max(conv_up2);
    cpadj = cp1max/cp2max;
    %conv_up2 = conv_up2.*cpadj*2.8;
    
    
    % create field noise for this timestep
    noise_u = n_u * randn(1, fieldSize);
    noise_p = n_p * randn(1, fieldSize);
    
    if w_n > 0 % create spatially correlated noise by convolution IS THIS USED?
      noise_u_padded = [noise_u(halfField+2:fieldSize), noise_u, noise_u(:, 1:halfField)];
      noise_u = conv2(1, kernel_q, noise_u_padded, 'valid');
      noise_p_padded = [noise_p(halfField+2:fieldSize), noise_p, noise_p(:, 1:halfField)];
      noise_p = conv2(1, kernel_q, noise_p_padded, 'valid');
    end
    
    % update field activities
    field_u = field_u + 1/time_u * (-field_u + h_u + conv_up1 + conv_up2 + conv_uu) + noise_u; % use +conv_uu ??
    field_P1 = field_P1 + 1/time_p1_build * (-field_P1 + h_pb +stimulusN1(t, :)) + noise_p; %+ conv_pp1
    field_P2 = field_P2 + 1/pop(N,5) * (-field_P2 + h_ps + stimulusN2(t, :)) + noise_p; %+ conv_pp2
    
   
    % store field activities at the selected time steps
    if any(tStoreFieldsEST == t)
      
      %history_n1(iHistory, :) = stimulusN1(t, :);
      %history_n2(iHistory, :) = stimulusN2(t,:);
      
      history_uEST(iHistory, :) = field_u;
      history_p1EST(iHistory, :) = field_P1;
      history_p2EST(iHistory, :) = field_P2;
      
      
      iHistory = iHistory + 1;
    end
    
    
     
        if 0
        indexP1 = find(field_P1(1,:)==max(field_P1(1,:))); %finds location of peak
        indexP2 = find(field_P2(1,:)==max(field_P2(1,:))); %finds location of peak
        peakP1 = max(field_P1(1,:));
        peakP2 = max(field_P2(1,:));
     
        %s_excPS1 = P1con * (1+ 1*indexP1/181);
        %s_excPS2 = P2con * (indexP2/indexP1); %the bigger value?
        end;
    
        
end
  %END OF TIME STEP LOOP
  
  
for c = 1:tMax;
      [MU,Q] = max(history_uEST((c+500*(i-1)),:)); %if all Inf it just reports index of first
      [MP,R] = max(history_p1EST((c+500*(i-1)),:));
      [MP2,S] = max(history_p2EST((c+500*(i-1)),:));
      actsumEST(c,i,N) = Q;
      actsump1EST(c,i,N) = R;
      actsump2EST(c,i,N) = S;
      
  end

%1. Determining the Desision from the P1 layer EST
uwindow = zeros(30,1); %defines window of recent 10 time steps for U layer
 
 for a = (tStimulusStart+50):500
 uwindow(1:30,1) = actsump1EST(a-30:a-1,i,N); %if current values deviation from each of the last 10 is less than 3 then set.

for d = 1:30
 uwindow(d,1) = abs(uwindow(d,1) - actsump1EST(a,i,N));
 end
 
 if max(uwindow(1:30,:)) < 3
     utime = a;
     udescision = actsump1EST(a,i,N);
     break
 end
 end
 
 
% 2. compariing model response to human responses EST
modeldataEST(i+(subtrialsEST*(b-1)),1,N) = N; %subject number in batch 1 - 10
modeldataEST(i+(subtrialsEST*(b-1)),2,N) = udescision; % %model estimate
modeldataEST(i+(subtrialsEST*(b-1)),3,N) = udescision - esthumandata(i,5); 
modeldataEST(i+(subtrialsEST*(b-1)),4,N) = abs(udescision - esthumandata(i,5)); %error from Human
modeldataEST(i+(subtrialsEST*(b-1)),5,N) = esthumandata(i,6); %TARGET VALUE
modeldataEST(i+(subtrialsEST*(b-1)),6,N) = abs(udescision - esthumandata(i,6)); %error from Target
modeldataEST(i+(subtrialsEST*(b-1)),7,N) = b;





 
 field_u(1:fieldSize) = h_u;
 field_P1(1:fieldSize) = h_pb;
 field_P2(1:fieldSize) = h_ps;
  
 field_N1(1:fieldSize) = h_p;
 field_A1(1:fieldSize) = h_p;
 field_D1(1:fieldSize) = h_p;
  
 field_N2(1:fieldSize) = h_p;
 field_A2(1:fieldSize) = h_p;
 field_D2(1:fieldSize) = h_p;


 currenttrial = currenttrial+1; %trial counter
end
%%%%%%%%%%%%%%%%
% END EST TRIALS LOOP
%%%%%%%%%%%%%%%%


%Set each of 10 subjects data here wihle in Subject Loop
%set U, P1, P2 layer data for all 10 subs
%this is used for vizualization. keep this the same for ANS task
data_u(:,:,N) = history_u;
data_P1(:,:,N) = history_p1;
data_P2(:,:,N) = history_p2;
data_N1(:,:,N) = history_n1;
%data_A1(:,:,N) = history_a1;
data_N2(:,:,N) = history_n2;
data_A2(:,:,N) = history_a2;
data_P1EST(:,:,N) = history_p1EST;
%data_(:,:,N) = history_d2;

 
%check human data, Descision and RT
%check last model trial data, Decision and RT
%note decision & RT for human and models. and difference
% i is the trial number
for g=1:subtrials
deviations(g,1,N) = tbthumandata(mod(g-1,subtrials)+1,9); % Human correctness
deviations(g,2,N) = tbthumandata(mod(g-1,subtrials)+1,3); %  Human RT

deviations(g,3:4,N) = modeldata(g+((b-1)*subtrials),1:2,N); % Model correctness, RT
deviations(g,5,N) = abs(deviations(g,1,N) - deviations(g,3,N)); % 0 matchtohuman, 1 nonmatch. ANS ERROR  

deviations(g,6,N) = (deviations(g,2,N) - deviations(g,4,N))^2; % 0 match, - fast, + slow
deviations(g,7,N) = deviations(g,1,N) + deviations(g,3,N) - 1; % -1 Incorrect match, 1 correct match, 0 non-match
end     

 %check human EST data, Descision and RT
%check last model trial data, Decision and RT
%note decision & RT for human and models. and difference
% i is the trial number
for g=1:subtrialsEST
deviationsEST(g,1,N) = esthumandata(mod(g-1,subtrialsEST)+1,10); % estimates,
deviationsEST(g,2,N) = modeldataEST(g+((b-1)*subtrialsEST),4,N); % abs deviation of EST from human
deviationsEST(g,3,N) = modeldataEST(g+((b-1)*subtrialsEST),6,N); % abs deviation of EST from Target
%deviationsEST(g,5,N) = abs(deviationsEST(g,1,N) - deviationsEST(g,3,N)); % 0 matchtohuman, 1 nonmatch  
%deviationsEST(g,6,N) = (deviationsEST(g,2,N) - deviationsEST(g,4,N))^2; % 0 match, - fast, + slow
%deviationsEST(g,7,N) = deviationsEST(g,1,N) + deviationsEST(g,3,N) - 1; % -1 Incorrect match, 1 correct match, 0 non-match
end     
%Should end up with matrices, modeldata and modelpop that have summary of all 10 subjects performance
%pass those to Evol Algo. 
        



end % SUBJECTS END





%%%%%%%%%%%%%%%
% PART 5.  EVOL ALGO
%%%%%%%%%%%%%%%

%rank the 10 subjects by fit. keep top 2. creat 8 new 

%look at deviations by N. to determine which of the 10 are worth keeping
modelfits = zeros(4,10);
RTvar = zeros(1,10);
percs = zeros(40,1);

% 5.1 DEFINE FITNESS

for z = 1:10
   %modelfits(1,z)= mean(deviations(:,5,z)); %decision error Higher is more error
   modelfits(1,z) = abs(mean(tbthumandata(:,9))- mean(deviations(:,3,z))); %ANS ERROr
   modelfits(2,z) = median(deviationsEST(:,2,z)); %EST human error
   modelfits(4,z) = (modelfits(1,z)*1) + (modelfits(2,z)*.01);
   %modelfits(3,z) = subtrials - sum(deviations(1:subtrials,3,z)); %(modelfits(1,z)*1) + (modelfits(2,z)*.1) ; % + (modelfits(2,z)/90) - RC(1,2) + (RTvar(1,z)*0) ; % GLOBAL ERROR
 
   %calculate precision
   %{
   for y = 1:5
   percs(y,1) = std(modeldataEST((8*y-7):(y*8),2,z))/esthumandata((8*y-7),6);
   end
    modelfits(3,z) = mean(percs(1:5,:)); % model EST PRECISION RATIO SCORE
   %}
   
   for y = 1:40
       percs(y,1) = abs(modeldataEST(y,2,z)-esthumandata(y,6))/(esthumandata(y,6));
   end
    modelfits(3,z) = mean(percs(1:40,1));

end
modelpop = pop;


% now modelpop has var for 10 subjects and last col is fitness

for r = 1:10 %subjects per batch

totalpop(r+(10*(b-1)),13) = b; %batch
totalpop(r+(10*(b-1)),14) = mean(modeldata(1+(subtrials*(b-1)):1+(subtrials-1)+(subtrials*(b-1)),1,r )); % corrent ANS
totalpop(r+(10*(b-1)),15) = mean(modeldata(1+(subtrials*(b-1)):1+(subtrials-1)+(subtrials*(b-1)),2,r )); % mean RT
totalpop(r+(10*(b-1)),16) = std(modeldata(1+(subtrials*(b-1)):1+(subtrials-1)+(subtrials*(b-1)),2,r )); %STD of RT
RC = corrcoef((modeldata(1+(subtrials*(b-1)):1+(subtrials-1)+(subtrials*(b-1)),2,r )),tbthumandata(1:subtrials,3));
%totalpop(r+(10*(b-1)),17) = RC(1,2); %RT correlation
totalpop(r+(10*(b-1)),18) = SubID;
%modelfits(3,r) = modelfits(3,r) + (.2)*(1-(RC(1,2))); %time correlation error
modelpop(r,9:12) = modelfits(1:4,r);
totalpop(r+(10*(b-1)),1:12) = modelpop(r,1:12);
end



[Y,I]=sortrows(modelpop(:,12)); %sort by total Error
poprank=modelpop(I,:); %use the column indices from sort() to sort all columns of A.

%for poprank row's 1:5 are copied to new pop

%Define new population
pop = zeros(10,12);
pop(1:3,1:12) = poprank(1:3,1:12);


%rows 5:10 are made new randomly

for w = 4:10
pop(w,1) = time_u; %randi([1,100],1,1);
pop(w,2)= s_inh; %randi([1,300],1,1);
pop(w,3) = n_u; %randi([1,500])*.0005; 
pop(w,4) = b_u; %randi([11,50])*.1;    
pop(w,5) = time_p2_build; %randi([250,350]);
pop(w,6) = s_excPS2; %randi([10,90])*.1;
pop(w,7) = randi([3,40]); %w_inh
pop(w,8) = randi([150,300])*.01; %ntcw
end


if evoalgo == 1;
    pop(1:10,2) = s_inh;
    pop(1:10,7) = w_inh;
else
    pop(1:10,8) = ntcw;
    
end

% new pop matrix is ready

%writes best pop stats to file
gengraph(b,4) = poprank(1,12);
gengraph(b,3) = poprank(1,11);
gengraph(b,2) = poprank(1,10);
gengraph(b,1) = b;

end %BATCH END


%%%%%%%%%%%%%%%%%%%%
% PART 6 
%DATA COLLECTION AND ANALYSIS
%%%%%%%%%%%%%%%%%%%%

% FIND LOWEST TOTAL ERROR AND WRITE LINE TO MATRIX WITH SUB LOOP 3 AS ROW

[M,I] = min(totalpop);
bestruns(CurrentSub,:) = totalpop(I(1,12),:);
totalruns(((SubID-1)*batches*N+1):N*SubID*batches,:) = totalpop;



%%%% save poprank, modelfits, modeldata, deviations.
header1 = {'generation','descisionerror','timeerror','TotalError'};
header2 = {'time_u','s_inh','n_u','b_u','time_p2','ERPfactor','ErrorDesc','ErrorEST','Totalerror'...
    ,'NegMatchError','Batch','Correct','MeanTime','TimeSdev','Correl'};
header3 = {'correctness','time','time_u','batch','descision','stim1','stim2'};
header4 = {'time_u','s_inh','n_u','b_u','time_p2','S_excP2','W_inh','ntcw','ErrorDesc','PREC_ERORR','ErrorESTtarget','Totalerror'...
    ,'Batch','Correct','MeanTime','TimeSdev','Correl','Subject','blank','blank'};
header5 = {'Sub','estimate','deviation_hum','abs_dev_hum','deivation_target','abs_dev_t','batch'};

%fopen('/Users/richardprather/Dropbox/Rich/PratherLab/backup/Modeling/Matlab/tbtmodtrials.csv');
%cd('/Users/Richard/Dropbox/Rich/PratherLab/backup/Modeling/Matlab')
dirwrite = pwd;

SubIDa = num2str(SubID);

if specB(2,1) == 2;

filelabelA = strcat('trialsANSD','_',SubIDa,'.csv');
filelabelB = strcat('trialsESTD','_',SubIDa,'.csv');
filelabelC = strcat('totalpopAED','_',SubIDa,'.csv');

csvwrite_with_headers(filelabelA,modeldata, header3);
csvwrite_with_headers(filelabelB,modeldataEST, header5);    

%csvwrite_with_headers('gengraphtest.csv',gengraph, header1); 
csvwrite_with_headers(filelabelC,totalpop, header4);

else
filelabelA = strcat('trialsANSP','_',SubIDa,'.csv');
filelabelB = strcat('trialsESTP','_',SubIDa,'.csv');
filelabelC = strcat('totalpopAEP','_',SubIDa,'.csv');    
csvwrite_with_headers(filelabelA,modeldata, header3);
csvwrite_with_headers(filelabelB,modeldataEST, header5);    

%csvwrite_with_headers('gengraphEST.csv',gengraph, header1); 
csvwrite_with_headers(filelabelC,totalpop, header4);




end;
% how well the model did is best summarized by deviations whenre each page
% is a model col1:2 is model data, col3:4 human, col5:6 is deviation

SubID = SubID+1;
end;
% END SUBJECT LOOP HERE

%EXPORT BESTRUN MATRIX TO CSV FILE

if specB(2,1) == 2;
csvwrite_with_headers('bestrunsD.csv',bestruns, header4);
csvwrite_with_headers('TotalrunsD.csv',totalruns, header4);
else
csvwrite_with_headers('bestrunsP.csv',bestruns, header4);
csvwrite_with_headers('TotalrunsP.csv',totalruns, header4);

end;
toc;
load handel
sound(y,Fs)









