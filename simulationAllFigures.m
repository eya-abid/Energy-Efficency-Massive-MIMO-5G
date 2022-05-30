clc;
close all;
clear all;

rng('shuffle');

Mmax = 220; %maximum number of antennas
Kmax = 150; %maximum number of UE (user equipment)






d_max = 250; %max cell radius
d_min = 35; %distance between the UEs and the BS
areaSinglecell = pi*(d_max/1000).^2; 
areaMulticell = 4*(d_max/1000).^2; 


dbar = 10^(-3.53);  
kappa = 3.76; 


B = 20e6; 
Bc = 180e3; 
Tc = 10e-3; 
U = Bc * Tc; 
sigma2B = 10^(-9.6-3); 


zetaDL = 0.6; 
zetaUL = 0.4; 


tauDL = 1; 
tauUL = 1; 


etaDL = 0.39; 
etaUL = 0.3; 
L_BS = 12.8e9; 
L_UE = 5e9; 
P_FIX = 18; 
P_SYN = 2; 
P_BS = 1; 
P_UE = 0.1; 
P_COD = 0.1e-9; 
P_DEC = 0.8e-9; 
P_BT = 0.25e-9; 


eta = 1/(zetaDL/etaDL + zetaUL/etaUL); 
S_x = (d_max^(kappa+2)-d_min^(kappa+2))/dbar/(1+kappa/2)/(d_max^2-d_min^2); 

Bsigma2SxetaSinglecell = sigma2B*S_x/eta; 
Bsigma2SxetaMulticell = 1.602212311888643; 







Ijl_PC_Reuse1 = [0.0975852990937329; 0.0975852990937329; 0.0975852990937329; 0.0975852990937329; 0.0237235488511305; 0.0237235488511305; 0.0237235488511305; 0.0237235488511305; 0.00429275445589164; 0.00429275445589164; 0.00429275445589164; 0.00429275445589164; 0.00276741832925105; 0.00276741832925105; 0.00276741832925105; 0.00276741832925105; 0.00276741832925105; 0.00276741832925105; 0.00276741832925105; 0.00276741832925105; 0.00106280112820823; 0.00106280112820823; 0.00106280112820823; 0.00106280112820823];
Ijl_PC_Reuse2 = [0.0237235488511305; 0.0237235488511305; 0.0237235488511305; 0.0237235488511305; 0.00429275445589164; 0.00429275445589164; 0.00429275445589164; 0.00429275445589164; 0.00106280112820823; 0.00106280112820823; 0.00106280112820823; 0.00106280112820823];
Ijl_PC_Reuse4 = [0.00429275445589164; 0.00429275445589164; 0.00429275445589164; 0.00429275445589164; 0.00106280112820823; 0.00106280112820823; 0.00106280112820823; 0.00106280112820823];



Ijl_nonPC_Reuse1 = [];
Ijl_nonPC_Reuse2 = [0.0975852990937329; 0.0975852990937329; 0.0975852990937329; 0.0975852990937329; 0.00276741832925105; 0.00276741832925105; 0.00276741832925105; 0.00276741832925105; 0.00276741832925105; 0.00276741832925105; 0.00276741832925105; 0.00276741832925105];
Ijl_nonPC_Reuse4 = [0.0975852990937329; 0.0975852990937329; 0.0975852990937329; 0.0975852990937329; 0.0237235488511305; 0.0237235488511305; 0.0237235488511305; 0.0237235488511305; 0.00276741832925105; 0.00276741832925105; 0.00276741832925105; 0.00276741832925105; 0.00276741832925105; 0.00276741832925105; 0.00276741832925105; 0.00276741832925105];


A = P_COD + P_DEC + P_BT; 
C_0 = P_FIX + P_SYN; 
C_1 = P_UE; 
C_2 = (4*B*tauDL)/(U*L_UE); 
D_0 = P_BS; 

C_3_ZF = B/(3*U*L_BS); 
D_1_ZF = B*(2+1/U)/L_BS; 
D_2_ZF = B*(3-2*tauDL)/(U*L_BS); 

C_3_MRT = 0; 
D_1_MRT = B*(2+3/U)/L_BS; 
D_2_MRT = B*(-2*tauDL)/(U*L_BS); 

Q = 3; 
C_3_MMSE = C_3_ZF*Q; 
D_1_MMSE = B*(2+Q*3/U)/L_BS; 
D_2_MMSE = B*(3*Q-2*tauDL)/(U*L_BS); 





EEoptZF = zeros(Mmax,Kmax); 
alphaOptsZF = zeros(Mmax,Kmax); 
sumRatesZF = NaN*ones(Mmax,Kmax); 
RFpowersZF = NaN*ones(Mmax,Kmax); 

EEoptMRT = zeros(Mmax,Kmax); 
sumRatesMRT = NaN*ones(Mmax,Kmax); 
RFpowersMRT = NaN*ones(Mmax,Kmax); 

EEoptMMSE = zeros(Mmax,Kmax); 
sumRatesMMSE = NaN*ones(Mmax,Kmax); 
RFpowersMMSE = NaN*ones(Mmax,Kmax); 

EEoptZFimperfect = zeros(Mmax,Kmax);  
sumRatesZFimperfect = NaN*ones(Mmax,Kmax); 
RFpowersZFimperfect = NaN*ones(Mmax,Kmax); 



EEoptZFMulticellReuse1 = zeros(Mmax,Kmax);  
sumRatesZFMulticellReuse1 = NaN*ones(Mmax,Kmax); 
RFpowersZFMulticellReuse1 = NaN*ones(Mmax,Kmax); 

EEoptZFMulticellReuse2 = zeros(Mmax,Kmax);  
sumRatesZFMulticellReuse2 = NaN*ones(Mmax,Kmax); 
RFpowersZFMulticellReuse2 = NaN*ones(Mmax,Kmax); 

EEoptZFMulticellReuse4 = zeros(Mmax,Kmax);  
sumRatesZFMulticellReuse4 = NaN*ones(Mmax,Kmax); 
RFpowersZFMulticellReuse4 = NaN*ones(Mmax,Kmax); 



for M = 1:Mmax
    
    
    disp(['Current number of antennas: ' num2str(M) '. Maximum: ' num2str(Mmax)]);
    
    
    for K = 1:Kmax
        
        
        
        if M>K
            
            
            Cprim = (C_0 + C_1*K + C_2*K^2 + C_3_ZF*K^3)/K;
            Dprim = (D_0 + D_1_ZF*K + D_2_ZF*K^2)/K;
            
            
            
            alphaOptsZF(M,K) = (exp(lambertw( (M-K)*(Cprim+M*Dprim)/exp(1)/Bsigma2SxetaSinglecell - 1/exp(1) ) +1 ) -1) / (M-K); 
            RFpowersZF(M,K) = alphaOptsZF(M,K)*K*Bsigma2SxetaSinglecell; 
            sumRatesZF(M,K) = B*K*(1-(tauUL+tauDL)*K/U)*log2(1+alphaOptsZF(M,K)*(M-K)); 
            EEoptZF(M,K) = ( sumRatesZF(M,K) ) / (  RFpowersZF(M,K) + C_0 + C_1*K + C_2*K^2 + C_3_ZF*K^3 + D_0*M + D_1_ZF*K*M + D_2_ZF*K^2*M + A*sumRatesZF(M,K) ); 
            
            
            
            
            prelogFactor = B*K*(1-(tauUL+tauDL)*K/U); 
            circuitpowerZF = C_0 + C_1*K + C_2*K^2 + C_3_ZF*K^3 + D_0*M + D_1_ZF*K*M + D_2_ZF*K^2*M; 
            
            
            
            alphaZFimp = fminsearch(@(x) -functionCalculateEEMulticellZF(x,M,K,[],[],tauUL,Bsigma2SxetaSinglecell,A,circuitpowerZF,prelogFactor)/1e6, alphaOptsZF(M,K)/2);
            [EEvalue,sumrateSinglecell,averageRFPower] = functionCalculateEEMulticellZF(alphaZFimp,M,K,[],[],tauUL,Bsigma2SxetaSinglecell,A,circuitpowerZF,prelogFactor);
            
            
            sumRatesZFimperfect(M,K) = sumrateSinglecell;
            EEoptZFimperfect(M,K) = EEvalue;
            RFpowersZFimperfect(M,K) = averageRFPower;
            
            
            
            
            prelogChannelReuse1 = B*K*(1-(tauUL+tauDL)*K/U); 
            
            
            
            alphaZFMulticell = fminsearch(@(x) -functionCalculateEEMulticellZF(x,M,K,Ijl_PC_Reuse1,Ijl_nonPC_Reuse1,tauUL,Bsigma2SxetaMulticell,A,circuitpowerZF,prelogChannelReuse1)/1e6, alphaOptsZF(M,K)/2);
            [EEvalue,rateMulticell,averageRFPower] = functionCalculateEEMulticellZF(alphaZFMulticell,M,K,Ijl_PC_Reuse1,Ijl_nonPC_Reuse1,tauUL,Bsigma2SxetaMulticell,A,circuitpowerZF,prelogChannelReuse1);
            
            
            sumRatesZFMulticellReuse1(M,K) = rateMulticell;
            EEoptZFMulticellReuse1(M,K) = EEvalue;
            RFpowersZFMulticellReuse1(M,K) = averageRFPower;
            
            
            
            
            prelogChannelReuse2 = B*K*(1-(2*tauUL+tauDL)*K/U); 
            
            
            
            alphaZFMulticell = fminsearch(@(x) -functionCalculateEEMulticellZF(x,M,K,Ijl_PC_Reuse2,Ijl_nonPC_Reuse2,2*tauUL,Bsigma2SxetaMulticell,A,circuitpowerZF,prelogChannelReuse2)/1e6, alphaOptsZF(M,K)/2);
            [EEvalue,rateMulticell,averageRFPower] = functionCalculateEEMulticellZF(alphaZFMulticell,M,K,Ijl_PC_Reuse2,Ijl_nonPC_Reuse2,2*tauUL,Bsigma2SxetaMulticell,A,circuitpowerZF,prelogChannelReuse2);
            
            
            sumRatesZFMulticellReuse2(M,K) = rateMulticell;
            EEoptZFMulticellReuse2(M,K) = EEvalue;
            RFpowersZFMulticellReuse2(M,K) = averageRFPower;
            
            
            
            
            prelogChannelReuse4 = B*K*(1-(4*tauUL+tauDL)*K/U); 
            
            
            
            alphaZFMulticell = fminsearch(@(x) -functionCalculateEEMulticellZF(x,M,K,Ijl_PC_Reuse4,Ijl_nonPC_Reuse4,4*tauUL,Bsigma2SxetaMulticell,A,circuitpowerZF,prelogChannelReuse4)/1e6, alphaOptsZF(M,K)/2);
            [EEvalue,rateMulticell,averageRFPower] = functionCalculateEEMulticellZF(alphaZFMulticell,M,K,Ijl_PC_Reuse4,Ijl_nonPC_Reuse4,4*tauUL,Bsigma2SxetaMulticell,A,circuitpowerZF,prelogChannelReuse4);
            
            
            sumRatesZFMulticellReuse4(M,K) = rateMulticell;
            EEoptZFMulticellReuse4(M,K) = EEvalue;
            RFpowersZFMulticellReuse4(M,K) = averageRFPower;
            
        end
        
        
      
            
            

    end
end










gridDensity = 25;




figure(3); hold on; box on;
title('Figure 3: ZF processing, Single-cell, Perfect CSI')

surface(1:Kmax,1:Mmax,EEoptZF/1e6,'EdgeColor','none'); 
colormap(autumn);


[EEvalues,indM] = max(EEoptZF,[],2);
[EEoptimal,indK] = max(EEvalues);
plot3(indM(indK),indK,EEoptimal/1e6,'k*','MarkerSize',10);




for m = [1 gridDensity:gridDensity:Mmax]
    plot3(1:Kmax,m*ones(1,Kmax),EEoptZF(m,:)/1e6,'k-');
end

for k = [1 gridDensity:gridDensity:Kmax]
    plot3(k*ones(1,Mmax),1:Mmax,EEoptZF(:,k)/1e6,'k-');
end

plot3(1:Kmax,1:Kmax,zeros(Kmax,1),'k-');

view([-46 24]);
axis([0 Kmax 0 Mmax 0 35]);

ylabel('Number of Antennas (M)');
xlabel('Number of Users (K)');
zlabel('Energy Efficiency [Mbit/Joule]');










figure(6); hold on; box on;
title('Figure 6: ZF processing, Single-cell, Imperfect CSI')

surface(1:Kmax,1:Mmax,EEoptZFimperfect/1e6,'EdgeColor','none'); 
colormap(autumn);


[EEvalues,indM] = max(EEoptZFimperfect,[],2);
[EEoptimal,indK] = max(EEvalues);
plot3(indM(indK),indK,EEoptimal/1e6,'k*','MarkerSize',10);


for m = [1 gridDensity:gridDensity:Mmax]
    plot3(1:Kmax,m*ones(1,Kmax),EEoptZFimperfect(m,:)/1e6,'k-');
end

for k = [1 gridDensity:gridDensity:Kmax]
    plot3(k*ones(1,Mmax),1:Mmax,EEoptZFimperfect(:,k)/1e6,'k-');
end

plot3(1:Kmax,1:Kmax,zeros(Kmax,1),'k-');

view([-46 24])
axis([0 Kmax 0 Mmax 0 30])

ylabel('Number of Antennas (M)');
xlabel('Number of Users (K)');
zlabel('Energy Efficiency [Mbit/Joule]');






Mrange = (1:Mmax)'; 



[~,optKzf] = max(EEoptZF,[],2);
[~,optKmrt] = max(EEoptMRT,[],2);
[~,optKmmse] = max(EEoptMMSE,[],2);
[~,optKzfimperfect] = max(EEoptZFimperfect,[],2);



optEEsZF = zeros(Mmax,1);
optEEsMRT = zeros(Mmax,1);
optEEsMMSE = zeros(Mmax,1);
optEEsZFimperfect = zeros(Mmax,1);



optRFpowersZF = zeros(Mmax,1);
optRFpowersMRT = zeros(Mmax,1);
optRFpowersMMSE = zeros(Mmax,1);
optRFpowersZFimperfect = zeros(Mmax,1);



optEEsumratesZF = zeros(Mmax,1);
optEEsumratesMRT = zeros(Mmax,1);
optEEsumratesMMSE = zeros(Mmax,1);
optEEsumratesZFimperfect = zeros(Mmax,1);


for M = 1:Mmax
    
    
    optEEsZF(M) = EEoptZF(M,optKzf(M))/1e6;
    optEEsMRT(M) = EEoptMRT(M,optKmrt(M))/1e6;
    optEEsMMSE(M) = EEoptMMSE(M,optKmmse(M))/1e6;
    optEEsZFimperfect(M) = EEoptZFimperfect(M,optKzfimperfect(M))/1e6;
    
    
    optRFpowersZF(M) = RFpowersZF(M,optKzf(M));
    optRFpowersMRT(M) = RFpowersMRT(M,optKmrt(M));
    optRFpowersMMSE(M) = RFpowersMMSE(M,optKmmse(M));
    optRFpowersZFimperfect(M) = RFpowersZFimperfect(M,optKzfimperfect(M));
    
    
    optEEsumratesZF(M) = sumRatesZF(M,optKzf(M))/1e9;
    optEEsumratesMRT(M) = sumRatesMRT(M,optKmrt(M))/1e9;
    optEEsumratesMMSE(M) = sumRatesMMSE(M,optKmmse(M))/1e9;
    optEEsumratesZFimperfect(M) = sumRatesZFimperfect(M,optKzfimperfect(M))/1e9;
    
end



[~,MoptimalZF] = max(optEEsZF);
[~,MoptimalMRT] = max(optEEsMRT);
[~,MoptimalMMSE] = max(optEEsMMSE);
[~,MoptimalZFimperfect] = max(optEEsZFimperfect);




figure(7); hold on; box on;
title('Figure 7: Single-cell, Comparison of EE values');

plot(Mrange,optEEsMMSE,'r-.','LineWidth',1);
plot(Mrange,optEEsZF,'k','LineWidth',1);
plot(Mrange,optEEsZFimperfect,'k:','LineWidth',1);
plot(Mrange,optEEsMRT,'b--','LineWidth',1);

plot(MoptimalMMSE,optEEsMMSE(MoptimalMMSE),'ro','LineWidth',1);
plot(MoptimalZF,optEEsZF(MoptimalZF),'ko','LineWidth',1);
plot(MoptimalZFimperfect,optEEsZFimperfect(MoptimalZFimperfect),'ko','LineWidth',1);
plot(MoptimalMRT,optEEsMRT(MoptimalMRT),'bo','LineWidth',1);

axis([0 Mmax 0 35]);

legend('MMSE (Perfect CSI)','ZF (Perfect CSI)','ZF (Imperfect CSI)','MRT (Perfect CSI)','Location','Best')

xlabel('Number of Antennas (M)');
ylabel('Energy Efficiency [Mbit/Joule]');





figure(8); hold on; box on;
title('Figure 8: Single-cell, Comparison of RF power and radiated power/antenna');

plot(Mrange,optRFpowersMMSE,'r-.','LineWidth',1);
plot(Mrange,optRFpowersZF,'k','LineWidth',1);
plot(Mrange,optRFpowersZFimperfect,'k:','LineWidth',1);
plot(Mrange,optRFpowersMRT,'b--','LineWidth',1);

plot(MoptimalMMSE,optRFpowersMMSE(MoptimalMMSE),'ro','LineWidth',1);
plot(MoptimalZF,optRFpowersZF(MoptimalZF),'ko','LineWidth',1);
plot(MoptimalZFimperfect,optRFpowersZFimperfect(MoptimalZFimperfect),'ko','LineWidth',1);
plot(MoptimalMRT,optRFpowersMRT(MoptimalMRT),'bo','LineWidth',1);

plot(Mrange,zetaDL*eta*optRFpowersMMSE./Mrange,'r-.','LineWidth',1);
plot(Mrange,zetaDL*eta*optRFpowersZF./Mrange,'k','LineWidth',1);
plot(Mrange,zetaDL*eta*optRFpowersZFimperfect./Mrange,'k:','LineWidth',1);
plot(Mrange,zetaDL*eta*optRFpowersMRT./Mrange,'b--','LineWidth',1);

plot(MoptimalMMSE,zetaDL*eta*optRFpowersMMSE(MoptimalMMSE)./MoptimalMMSE,'ro','LineWidth',1);
plot(MoptimalZF,zetaDL*eta*optRFpowersZF(MoptimalZF)./MoptimalZF,'ko','LineWidth',1);
plot(MoptimalZFimperfect,zetaDL*eta*optRFpowersZFimperfect(MoptimalZFimperfect)./MoptimalZFimperfect,'ko','LineWidth',1);
plot(MoptimalMRT,zetaDL*eta*optRFpowersMRT(MoptimalMRT)./MoptimalMRT,'bo','LineWidth',1);

set(gca,'YScale','Log');
axis([0 Mmax 1e-2 1e2]);

text(35,4.5,'Total RF power');
text(35,0.25,'Radiated power per BS antenna');
legend('MMSE (Perfect CSI)','ZF (Perfect CSI)','ZF (Imperfect CSI)','MRT (Perfect CSI)','Location','Best')

xlabel('Number of Antennas (M)');
ylabel('Average Power [W]');





figure(9); hold on; box on;
title('Figure 9: Single-cell, Comparison of area throughput');

plot(Mrange,optEEsumratesMMSE/areaSinglecell,'r-.','LineWidth',1);
plot(Mrange,optEEsumratesZF/areaSinglecell,'k','LineWidth',1);
plot(Mrange,optEEsumratesZFimperfect/areaSinglecell,'k:','LineWidth',1);
plot(Mrange,optEEsumratesMRT/areaSinglecell,'b--','LineWidth',1);

plot(MoptimalMMSE,optEEsumratesMMSE(MoptimalMMSE)/areaSinglecell,'ro','LineWidth',1);
plot(MoptimalZF,optEEsumratesZF(MoptimalZF)/areaSinglecell,'ko','LineWidth',1);
plot(MoptimalZFimperfect,optEEsumratesZFimperfect(MoptimalZFimperfect)/areaSinglecell,'ko','LineWidth',1);
plot(MoptimalMRT,optEEsumratesMRT(MoptimalMRT)/areaSinglecell,'bo','LineWidth',1);

axis([0 Mmax 0 70]);

legend('MMSE (Perfect CSI)','ZF (Perfect CSI)','ZF (Imperfect CSI)','MRT (Perfect CSI)','Location','NorthWest')

xlabel('Number of Antennas (M)');
ylabel('Area Throughput [Gbit/s/km^2]');







[~,optKMulticellReuse1] = max(EEoptZFMulticellReuse1,[],2);
[~,optKMulticellReuse2] = max(EEoptZFMulticellReuse2,[],2);
[~,optKMulticellReuse4] = max(EEoptZFMulticellReuse4,[],2);



optEEsMulticellReuse1 = zeros(Mmax,1);
optEEsMulticellReuse2 = zeros(Mmax,1);
optEEsMulticellReuse4 = zeros(Mmax,1);



optRFpowersMulticellReuse1 = zeros(Mmax,1);
optRFpowersMulticellReuse2 = zeros(Mmax,1);
optRFpowersMulticellReuse4 = zeros(Mmax,1);



optEEsumratesMulticellReuse1 = zeros(Mmax,1);
optEEsumratesMulticellReuse2 = zeros(Mmax,1);
optEEsumratesMulticellReuse4 = zeros(Mmax,1);


for M = 1:Mmax
    
    
    optEEsMulticellReuse1(M) = EEoptZFMulticellReuse1(M,optKMulticellReuse1(M))/1e6;
    optEEsMulticellReuse2(M) = EEoptZFMulticellReuse2(M,optKMulticellReuse2(M))/1e6;
    optEEsMulticellReuse4(M) = EEoptZFMulticellReuse4(M,optKMulticellReuse4(M))/1e6;
    
    
    optRFpowersMulticellReuse1(M) = RFpowersZFMulticellReuse1(M,optKMulticellReuse1(M));
    optRFpowersMulticellReuse2(M) = RFpowersZFMulticellReuse2(M,optKMulticellReuse2(M));
    optRFpowersMulticellReuse4(M) = RFpowersZFMulticellReuse4(M,optKMulticellReuse4(M));
    
    
    optEEsumratesMulticellReuse1(M) = sumRatesZFMulticellReuse1(M,optKMulticellReuse1(M))/1e9;
    optEEsumratesMulticellReuse2(M) = sumRatesZFMulticellReuse2(M,optKMulticellReuse2(M))/1e9;
    optEEsumratesMulticellReuse4(M) = sumRatesZFMulticellReuse4(M,optKMulticellReuse4(M))/1e9;
    
end



[~,MoptimalReuse1] = max(optEEsMulticellReuse1);
[~,MoptimalReuse2] = max(optEEsMulticellReuse2);
[~,MoptimalReuse4] = max(optEEsMulticellReuse4);




figure(11); hold on; box on;
title('Figure 11: Multi-cell, Comparison of EE values');

plot(Mrange,optEEsMulticellReuse4,'b--','LineWidth',1);
plot(Mrange,optEEsMulticellReuse2,'k','LineWidth',1);
plot(Mrange,optEEsMulticellReuse1,'r-.','LineWidth',1);

plot(MoptimalReuse4,optEEsMulticellReuse4(MoptimalReuse4),'bo','LineWidth',1);
plot(MoptimalReuse2,optEEsMulticellReuse2(MoptimalReuse2),'ko','LineWidth',1);
plot(MoptimalReuse1,optEEsMulticellReuse1(MoptimalReuse1),'ro','LineWidth',1);

axis([0 Mmax 0 8]);

legend('ZF (Imperfect CSI): Reuse 4','ZF (Imperfect CSI): Reuse 2','ZF (Imperfect CSI): Reuse 1','Location','SouthEast');

xlabel('Number of Antennas (M)');
ylabel('Energy Efficiency [Mbit/Joule]');





figure(12); hold on; box on;
title('Figure 12: Multi-cell, Comparison of RF power and radiated power/antenna')

plot(Mrange,optRFpowersMulticellReuse4,'b--','LineWidth',1);
plot(Mrange,optRFpowersMulticellReuse2,'k','LineWidth',1);
plot(Mrange,optRFpowersMulticellReuse1,'r-.','LineWidth',1);

plot(MoptimalReuse4,optRFpowersMulticellReuse4(MoptimalReuse4),'bo','LineWidth',1);
plot(MoptimalReuse2,optRFpowersMulticellReuse2(MoptimalReuse2),'ko','LineWidth',1);
plot(MoptimalReuse1,optRFpowersMulticellReuse1(MoptimalReuse1),'ro','LineWidth',1);


plot(Mrange,zetaDL*eta*optRFpowersMulticellReuse1./Mrange,'r-.','LineWidth',1);
plot(Mrange,zetaDL*eta*optRFpowersMulticellReuse2./Mrange,'k','LineWidth',1);
plot(Mrange,zetaDL*eta*optRFpowersMulticellReuse4./Mrange,'b--','LineWidth',1);

plot(MoptimalReuse4,zetaDL*eta*optRFpowersMulticellReuse4(MoptimalReuse4)./MoptimalReuse4,'bo','LineWidth',1);
plot(MoptimalReuse2,zetaDL*eta*optRFpowersMulticellReuse2(MoptimalReuse2)./MoptimalReuse2,'ko','LineWidth',1);
plot(MoptimalReuse1,zetaDL*eta*optRFpowersMulticellReuse1(MoptimalReuse1)./MoptimalReuse1,'ro','LineWidth',1);

set(gca,'YScale','Log');
axis([0 Mmax 1e-2 1e2]);

text(20,5.5,'Total RF power');
text(20,0.15,'Radiated power per BS antenna');
legend('ZF (Imperfect CSI): Reuse 4','ZF (Imperfect CSI): Reuse 2','ZF (Imperfect CSI): Reuse 1','Location','Best');

xlabel('Number of Antennas (M)');
ylabel('Average Power [W]');




figure(13); hold on; box on;
title('Figure 13: Multi-cell, Comparison of area throughput')

plot(Mrange,optEEsumratesMulticellReuse4/areaMulticell,'b--','LineWidth',1);
plot(Mrange,optEEsumratesMulticellReuse2/areaMulticell,'k','LineWidth',1);
plot(Mrange,optEEsumratesMulticellReuse1/areaMulticell,'r-.','LineWidth',1);

plot(MoptimalReuse4,optEEsumratesMulticellReuse4(MoptimalReuse4)/areaMulticell,'bo','LineWidth',1);
plot(MoptimalReuse2,optEEsumratesMulticellReuse2(MoptimalReuse2)/areaMulticell,'ko','LineWidth',1);
plot(MoptimalReuse1,optEEsumratesMulticellReuse1(MoptimalReuse1)/areaMulticell,'ro','LineWidth',1);

axis([0 Mmax 0 9]);

legend('ZF (Imperfect CSI): Reuse 4','ZF (Imperfect CSI): Reuse 2','ZF (Imperfect CSI): Reuse 1','Location','NorthWest')

xlabel('Number of Antennas (M) ');
ylabel('Area Throughput [Gbit/s/km^2]');





figure(14); grid on; hold on;
title('Figure 14: ZF processing, Multi-cell, Pilot reuse 4')

surface(1:Kmax,1:Mmax,EEoptZFMulticellReuse4(1:Mmax,:)/1e6,'EdgeColor','none'); 
colormap(autumn);


[EEvalues,indM] = max(EEoptZFMulticellReuse4,[],2);
[EEoptimal,indK] = max(EEvalues);
plot3(indM(indK),indK,EEoptimal/1e6,'k*','MarkerSize',10);


for m = [1 gridDensity:gridDensity:Mmax]
    plot3(1:Kmax,m*ones(1,Kmax),EEoptZFMulticellReuse4(m,:)/1e6,'k-');
end

for k = [1 gridDensity:gridDensity:Kmax]
    plot3(k*ones(1,Mmax),1:Mmax,EEoptZFMulticellReuse4(:,k)/1e6,'k-');
end

plot3(1:Kmax,1:Kmax,zeros(Kmax,1),'k-');

view([-46 24])
axis([0 Kmax 0 Mmax 0 8])

ylabel('Number of Antennas (M)');
xlabel('Number of Users (K)');
zlabel('Energy Efficiency [Mbit/Joule]');
