%load('nfdata.mat') % input any of the data files
%results column 1 = number of neurons
%results column 2 = average F1
%results column 3 = average Recall
%results column 4 = average Precision
%results1 = SAND, results2 = SL+FLHO, results3 = SUNS
%load('nffpr.mat') % CaImAn and Suite2p results

figure()
txt = 'YST' %Plot Title
xlimits = [0 300];
n = 50; %bin width
t = 0:n:300; %x range
r1 = zeros(7,length(t));
r2 = zeros(7,length(t));
r3 = zeros(7,length(t));


for i = 1:length(t)
    res1 = results(results(:,1)>t(i),:);
    res1 = res1(res1(:,1)<=t(i)+n,:);
    r1(1,i) = nanmean(res1(:,2));
    r1(2,i) = nanstd(res1(:,2))./sqrt(length(res1(:,2)));
    r1(3,i) = nanmean(res1(:,3));
    r1(4,i) = nanstd(res1(:,3))./sqrt(length(res1(:,3)));
    r1(5,i) = nanmean(res1(:,4));
    r1(6,i) = nanstd(res1(:,4))./sqrt(length(res1(:,4)));    
    r1(7,i) = length(res1(:,2))
   
    res2 = results2(results2(:,1)>t(i),:);
    res2 = res2(res2(:,1)<=t(i)+n,:);
    r2(1,i) = nanmean(res2(:,2));
    r2(2,i) = nanstd(res2(:,2))./sqrt(length(res2(:,2)));
    r2(3,i) = nanmean(res2(:,3));
    r2(4,i) = nanstd(res2(:,3))./sqrt(length(res2(:,3)));
    r2(5,i) = nanmean(res2(:,4));
    r2(6,i) = nanstd(res2(:,4))./sqrt(length(res2(:,4)));  
    r2(7,i) = length(res2(:,2))

    res3 = results3(results3(:,1)>t(i),:);
    res3 = res3(res3(:,1)<=t(i)+n,:);
    r3(1,i) = nanmean(res3(:,2));
    r3(2,i) = nanstd(res3(:,2))./sqrt(length(res3(:,2)));
    r3(3,i) = nanmean(res3(:,3));
    r3(4,i) = nanstd(res3(:,3))./sqrt(length(res3(:,3)));
    r3(5,i) = nanmean(res3(:,4));
    r3(6,i) = nanstd(res3(:,4))./sqrt(length(res3(:,4)));
    r3(7,i) = length(res3(:,2))

end

subplot(1,3,1)
x=5; %scatter plot marker size
a = 1; %scatter plot marker alpha
scatter(results3(:,1),results3(:,2),x,[0.9804    0.8275    0.6157],'filled','MarkerFaceAlpha',a,'MarkerEdgeAlpha',a)
hold on;
scatter(results2(:,1),results2(:,2),x,[ 0.7098    0.7922    1.0000],'filled','MarkerFaceAlpha',a,'MarkerEdgeAlpha',a)
scatter(results(:,1),results(:,2),x,[ 1.0000    0.7686    0.7882],'filled','MarkerFaceAlpha',a,'MarkerEdgeAlpha',a)
title('F1')
subplot(1,3,2)
scatter(results3(:,1),results3(:,3),x,[0.9804    0.8275    0.6157],'filled','MarkerFaceAlpha',a,'MarkerEdgeAlpha',a)
hold on;
scatter(results2(:,1),results2(:,3),x,[0.7098    0.7922    1.0000],'filled','MarkerFaceAlpha',a,'MarkerEdgeAlpha',a)
scatter(results(:,1),results(:,3),x,[ 1.0000    0.7686    0.7882],'filled','MarkerFaceAlpha',a,'MarkerEdgeAlpha',a)
title('Recall')
subplot(1,3,3)
scatter(results3(:,1),results3(:,4),x,[0.9804    0.8275    0.6157],'filled','MarkerFaceAlpha',a,'MarkerEdgeAlpha',a)
hold on;
scatter(results2(:,1),results2(:,4),x,[ 0.7098    0.7922    1.0000],'filled','MarkerFaceAlpha',a,'MarkerEdgeAlpha',a)
scatter(results(:,1),results(:,4),x,[1.0000    0.7686    0.7882],'filled','MarkerFaceAlpha',a,'MarkerEdgeAlpha',a)
title('Precision')

subplot(1,3,1)
s = shadedErrorBar(t+n/2,r1(1,:),r1(2,:),'lineprops',{'color',[0.6353    0.0784    0.1843]})
set(s.edge,'LineWidth',2,'LineStyle','none')
hold on;
s = shadedErrorBar(t+n/2,r2(1,:),r2(2,:),'lineprops',{'color',[0.0    0.324    0.9098]})
set(s.edge,'LineWidth',2,'LineStyle','none')
s = shadedErrorBar(t+n/2,r3(1,:),r3(2,:),'lineprops',{'color',[0.8392    0.5255    0.1922]})
set(s.edge,'LineWidth',2,'LineStyle','none')
yl = yline(f1caiman,'-')
yl.Color = [.87 .72 .95];
yl2 = yline(f1suite2p,'--')
yl2.Color = [.87 .72 .95];
ylim([0 1])
xlim(xlimits)
title('F1')
subplot(1,3,2)
s = shadedErrorBar(t+n/2,r1(3,:),r1(4,:),'lineprops',{'color',[0.6353    0.0784    0.1843]})
set(s.edge,'LineWidth',2,'LineStyle','none')
hold on;
s = shadedErrorBar(t+n/2,r2(3,:),r2(4,:),'lineprops',{'color',[0.0    0.324    0.9098]})
set(s.edge,'LineWidth',2,'LineStyle','none')
s = shadedErrorBar(t+n/2,r3(3,:),r3(4,:),'lineprops',{'color',[0.8392    0.5255    0.1922]})
set(s.edge,'LineWidth',2,'LineStyle','none')
yl = yline(recallcaiman,'-')
yl.Color = [.87 .72 .95];
yl2 = yline(recallsuite2p,'--')
yl2.Color = [.87 .72 .95];
ylim([0 1])
xlim(xlimits)
xlabel('Number of Neurons Trained On')
title('Recall')
subplot(1,3,3)
s = shadedErrorBar(t+n/2,r1(5,:),r1(6,:),'lineprops',{'color',[0.6353    0.0784    0.1843]})
set(s.edge,'LineWidth',2,'LineStyle','none')
hold on;
s = shadedErrorBar(t+n/2,r2(5,:),r2(6,:),'lineprops',{'color',[0.0    0.324    0.9098]})
set(s.edge,'LineWidth',2,'LineStyle','none')
s = shadedErrorBar(t+n/2,r3(5,:),r3(6,:),'lineprops',{'color',[0.8392    0.5255    0.1922]})
set(s.edge,'LineWidth',2,'LineStyle','none')
yl = yline(precisioncaiman,'-')
yl.Color = [.87 .72 .95];
yl2 = yline(precisionsuite2p,'--')
yl2.Color = [.87 .72 .95];
ylim([0 1])
xlim(xlimits)
title('Precision')
sgtitle(txt)
