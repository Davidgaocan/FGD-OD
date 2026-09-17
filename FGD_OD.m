function [r1_final,r2,area]=FGDOD(data_norm,lammda, kkk)
[n,m]=size(data_norm);
data = data_norm(:, 1 : m - 1);
m = m - 1;
%% Compute dist between i and j under all features
dist = zeros(n, m);
for ii = 1 : n
    dist(:, ii) = sum(abs(repmat(data(ii, :), n, 1) - data), 2) / m ;        
end
delta=zeros(1,m);% Initialize neighborhood radius
for j=1:m
    if min(data(:,j))==0&&max(data(:,j))==1
     delta(j)=std(data(:,j),1)/lammda; % compute neighborhood radius of numerical feature
    end
end
%% Compute sample similarity for each feature
num = 1;
for l=1:m
col=l;
r=zeros(n,n);
% eval(['NbrSet' num2str(col) '=zeros(n,n);']);
for j=1:n      
    a=data(j,col);
     x=data(:,col);       
         c = abs(a - x)';
         r(j,:)=ufrs_kersim(a,x,delta(l),n);
         r(:,j)=r(j,:);
end
if cal_E(r) ~= 0
eval(['NbrSet' num2str(num) '=r;']);
num = num + 1;
end
end
m = num - 1;
%% Compute sample similarity under all features
Simi_total = ones(n, n);
NH_total = zeros(m, 1);
for l = 1 : m
    eval(['Simi_total = min(Simi_total, NbrSet' num2str(l) ');'])
    NH_total(l) = cal_E(Simi_total);
end
ent_final = cal_E(Simi_total);
% 
%% Compute entropy
NH = zeros(m, 1);
for l = 1 : m
    eval(['data_tem = NbrSet' num2str(l) ';'])
    NH(l) = cal_E(data_tem);
end
%% Compute joint entropy and mutal information
joint_E = zeros(m, m);
MI = zeros(m, m);
for l1 = 1 : m
    eval(['data_tem_1 = NbrSet' num2str(l1) ';'])
    for l2 = 1 : m
        eval(['data_tem_2 = NbrSet' num2str(l2) ';'])
        data_min = min(data_tem_1, data_tem_2);
        joint_E(l1, l2) = cal_E(data_min);
        MI(l1, l2) = NH(l1) + NH(l2) - joint_E(l1, l2);
    end
    MI(l1, l1) = 0;
end
a = 1;
%% 
Su = 1 : m;     %unselected
MI_sum = sum(MI, 2);
% Su = 1:n;
max_tem = max(MI_sum);
num_tem = find(MI_sum == max_tem);
if(size(num_tem, 1) == 1)
    order(1) = Su(num_tem);
    Su(num_tem) = [];
else
    order(1) = Su(num_tem(1));
    Su(num_tem(1)) = [];
end
eval(['simi_select = NbrSet' num2str(order(1)) ';'])
mrmr = zeros(m ,1);

for l = 2 : m
    FRel = zeros(size(Su, 2),1);
    FRed = zeros(size(Su, 2),1);
    for ll = 1 : size(Su, 2)
    eval(['simi_select_tem = min(simi_select, NbrSet' num2str(Su(ll)) ');'])        
    FRel_temp = zeros(size(Su, 2),1);    
    for lll = 1 : size(Su, 2)
        %% FREL after adding a feature
        eval(['data_tem_2 = NbrSet' num2str(Su(lll)) ';'])
        data_tem_3 = min(data_tem_2, simi_select_tem);
        FRel_temp(lll) = (cal_E(data_tem_2) + cal_E(simi_select_tem) - cal_E(data_tem_3)) / cal_E(data_tem_2);         
    end
        FRel(ll) = sum(FRel_temp) / (size(Su, 2));
        eval(['sele_tem = NbrSet' num2str(Su(ll)) ';'])
        FRed_tem = zeros(size(order, 2), 1);
        data_tem_5 = min(simi_select, sele_tem);
        FRed(ll) = (cal_E(simi_select) + cal_E(sele_tem) - cal_E(data_tem_5)) / cal_E(sele_tem);
    end

    RR = FRel - FRed;
    mrmr(l) = max(RR);
    max_tem = max(RR);
    num_tem = find(RR == max_tem);
    if(size(num_tem, 1) == 1)
        order(size(order, 2) + 1) = Su(num_tem);
        Su(num_tem) = [];
    else
        ent_tem = zeros(m, 1);
        ent_tem(num_tem) = NH(Su(num_tem));
        a = find(ent_tem == max(ent_tem(num_tem)));
        tem = a(1);
        order(size(order, 2) + 1) = Su(tem);
        Su(num_tem(1)) = [];
    end
    eval(['simi_select = min(simi_select, NbrSet' num2str(order(size(order, 2))) ');'])
end

%% Compute information coverage
simi_select = ones(1, 1);
for l = 1 : size(order, 2)
        eval(['simi_select = min(simi_select, NbrSet' num2str(order(l)) ');'])        
        con_temp = zeros(m,1);    
        for ll = 1 : m
            eval(['data_tem_2 = NbrSet' num2str(ll) ';'])
            data_tem_3 = min(data_tem_2, simi_select);
            con_temp(ll) = (cal_E(data_tem_2) + cal_E(simi_select) - cal_E(data_tem_3)) / cal_E(data_tem_2);
        end
        MI_avg(l) = sum(con_temp) / m; 
end
MI_sub = zeros(m, 1);
for l = 2 : m
    MI_sub(l) = MI_avg(l) - MI_avg(l - 1);
end
%% ascending
eval(['simi_sum1 = NbrSet' num2str(order(1)) ';'])
for l = 2 : size(order, 1)
    eval(['simi_sum' num2str(l) '= simi_sum' num2str(l - 1) '+ NbrSet' num2str(order(l)) ';'])
end
for l = 1 : size(order, 1)
    eval(['simi_sum' num2str(l) ' = simi_sum' num2str(l) ' / l;'])
end

sigma = 0.1 : 0.1 : 2;
area = zeros(size(sigma, 2), 1);

%% Compute max
for kk = 1 : size(sigma, 2)
    kk 
    rel_gra_max = zeros(n, m);
    rel_gra_min = zeros(n, m);
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for l = 1 : m
    %%Find the k-dist
   eval([' att_sum = sum(NbrSet' num2str(l) ', 2) ;'])
         num1 = max(1, floor(sum(att_sum) / n * sigma(kk)));
         k = min(num1, n - 1);
         k_neighbor = zeros(k, n);
        % Find kNN
        for ii = 1 : n            
            eval(['simi_tem = NbrSet' num2str(l) '(ii, :) ;'])
            simi_tem_sort = sort(simi_tem, "descend");
            k_distance = simi_tem_sort(k + 1);
            k_neighbor_tem = find(simi_tem >= k_distance);
            k_neighbor_tem(find(k_neighbor_tem == ii)) = [];
            if size(k_neighbor_tem, 2) ~= k
                tem = att_sum(k_neighbor_tem);
                [tem, num_tem] = sort(tem, "descend");
                aa = tem(k);
                bbbb = k_neighbor_tem(num_tem(1:k));
                num_tem = find(att_sum(k_neighbor_tem) >= aa);
                k_neighbor(:, ii) = bbbb;
            else
                k_neighbor(:, ii) = k_neighbor_tem';
            end
            
        end
    %% Compute k relative density    
    for ii = 1 : n
       avg_gra_max(ii) = sum(max(repmat(att_sum(ii), k, 1), att_sum(k_neighbor(:, ii)))) / k; 
       avg_gra_min(ii) = sum(min(repmat(att_sum(ii), k, 1), att_sum(k_neighbor(:, ii)))) / k; 
    end
    for ii = 1 : n
        nei_max = sum(avg_gra_max(k_neighbor(:, ii)), 2) / k;
        rel_gra_max(ii, l) = nei_max; 
        nei_min = sum(avg_gra_min(k_neighbor(:, ii)), 2) / k;
        rel_gra_min(ii, l) = avg_gra_min(ii);  
        gra(ii, l) = avg_gra_min(ii) / avg_gra_max(ii);
    end
    for ii = 1 : n
     gra_rel2(ii, l) = gra(ii, l) / (sum(gra(k_neighbor(:, ii), l)) / k);
    end
    end
    rel_gra_rel2 = 1 - gra_rel2;
    rel_gra_rel2(find(rel_gra_rel2 < 0)) = 0;

    sigma_ent = 0.8 : 0.01 : 1;
    area_tem = zeros(m, 1);
    r1_tem = zeros(size(sigma_ent, 2), 1);
    area_ttem = zeros(21, 1);
      for ii = 1 : size(order, 2)
          if MI_avg(ii) < 0.8
              continue
          end
          
          MI_tem = fix(MI_avg(ii) * 100);
          if (area_ttem(MI_tem - 79) == 0)
            final(:, 1) = 1 : n;
            final(:, 2) = sum(rel_gra_rel2(:, order(1 : ii)) , 2);
        
            final = sortrows(final, -2);
            num = final(:, 1);
            degree = final(:, 2);
            [P1,R1,F1,r1_tem(ii),r2,area_tem(ii)]=outlier2_dection(data_norm,num); 
            area_ttem(MI_tem - 79) = area_tem(ii);
          end
    end
    area(kk) = max(area_ttem);
    num_tem = find(area_ttem == area(kk));
    r1(kk) = min(r1_tem(num_tem));
end
r1_final = 1;
end


function kersim=ufrs_kersim(a,x,e,n)
    kersim=zeros(1,n);
    if (e==0)
        temp1=(a==x);
        temp_1=find(temp1==1);
        temp_2=find(temp1==0);
        kersim(temp_1)=1;
        kersim(temp_2)=0;

    else
        temp1=abs(x - a)>e;
        temp_1=find(temp1==1);
        temp_2=find(temp1==0);
        kersim(temp_1)=0;
        kersim(temp_2)=(1-abs(x(temp_2) - a));
    end
end

function E = cal_E(data)
    [n, m] = size(data);
    data_sum = sum(data, 2) / n;
    a = log2(data_sum);
    a(isinf(a)) = 0;
    E = - sum(a) / n;
end


function Con_E = cal_con_E(data1, data2)
    data3 = min(data1, data2);
    Con_E = cal_E(data3) - cal_E(data2);
end