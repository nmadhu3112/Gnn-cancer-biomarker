tissues = {'Breast','Liver', 'Prostate','Bladder'};
for t = 1:length(tissues)
    tissue = tissues{t};

    % 1. load predict probabilities
    data = importdata(['data/HPA_cancer/Adjacent/' tissue '/gene_list']);
    % 2. calculate the similarity between normal and cancer condition
    scores = zeros(size(data,1),2);
    for i = 1:size(data,1)
        estimates =35;
        props_n = zeros(estimates,6);
        props_c = zeros(estimates,6);
        for esti = 1:estimates
            load(['data/HPA_cancer/Adjacent/' tissue '/normal/protein/est' num2str(esti-1) '/norProbs.mat'])
            load(['data/HPA_cancer/Adjacent/' tissue '/cancer/protein/est' num2str(esti-1) '/canProbs.mat'])
            props_n(esti,:) = norProbs(i,:);
            props_c(esti,:) = canProbs(i,:);
        end
        % euclidean distance and correlation coefficient
        euc = zeros(size(props_n,1)*size(props_c,1),1); % Euc 35*35
        cc = zeros(size(props_n,1)*size(props_c,1),1); % CC 35*35
        nd = 0;
        for j = 1:size(props_n,1)
            for k = 1:size(props_c,1)
                nd = nd+1;
                euc(nd,1) = sqrt(sum((props_n(j,:) - props_c(k,:)).^2)); % Euc pairs
                tmp = corrcoef([props_n(j,:); props_c(k,:)]');
                cc(nd,1) = tmp(1,2); % CC pairs
            end
        end
        % random background
        num = 100; % multiple hypothesis correction
        eucPvalues = zeros(num,1); % Euc p values
        ccPvalues = zeros(num,1); % CC p values
        for n = 1:num
            rdnum = 1000;
            g1 = zeros(rdnum,1);
            g2 = zeros(rdnum,1); 
            ne = 0;
            for j = 1:rdnum % random from cancer
                ind = randsample(size(props_c,1),2);
                ne = ne+1;
                g1(ne,1) = sqrt(sum((props_c(ind(1),:) - props_c(ind(2),:)).^2));
                tmp = corrcoef([props_c(ind(1),:); props_c(ind(2),:)]');
                g2(ne,1) = tmp(1,2);
            end  
            dg1 = [euc; g1];
            [dg1rank,r1] = sort(dg1,'descend'); % Euc
            [~,r1] = sort(r1);
            r1 = r1(1:length(euc),:);
            eucPvalues(n,1) = mean(r1)/length(dg1);
            dg2 = [cc; g2];
            [dg2rank,r2] = sort(dg2); % CC
            [~,r2] = sort(r2);
            r2 = r2(1:length(cc),:); 
            ccPvalues(n,1) = mean(r2)/length(dg2);
        end
        % euclidean distance
        scores(i,1) = mean(euc);
        % correlation coefficient
        scores(i,2) = mean(cc);
        % euc p value
        scores(i,3) = mean(eucPvalues);
        % cc p value
        scores(i,4) = mean(ccPvalues);
    end

    filename = ['p_values/' tissue '.txt'];
    save(filename, 'scores','-ascii');
end


