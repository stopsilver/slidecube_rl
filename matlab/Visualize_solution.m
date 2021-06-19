%% Visualize_solution_02.m

N=2;
[xcoord,ycoord,zcoord,rotplane,cuberotation_idx]=slidecube_init(N);

% t=xcoord;xcoord=ycoord;ycoord=t;

%load testsolution.mat
solmoves=load('..\solutionmoves.txt');
sol=ones(length(solmoves),2);
for i=1:length(solmoves)
	sol(i,1)=floor(solmoves(i)/(N*4-1))+1;
	sol(i,2)=mod(solmoves(i),(N*4-1))+1;
end

% rotplane=rotplane(:,[5 6 3 4 1 2]);

%startpos=ConvertToPos(razvertka_startpos,rotplane,N);
startpos=load('..\startpos.txt')+1;

% sclr=[
% 0 0 1; ...
% 0 1 0; ...
% 1 0 0; ...
% 1 0.5 0; ...
% 1 1 0; ...
% 0 0 0; ...
%     ];
sclr=[
1 1 0; ...
0 0 1; ...
1 0 0; ...
0 1 0; ...
1 0.75 0; ...
1 1 1; ...
    ];

pos=startpos;
c=zeros(length(xcoord),size(sclr,2));
for i=1:length(xcoord), c(i,:)=sclr(pos(i),:); end
figure(1)
h=scatter3(xcoord,ycoord,zcoord,3000*ones(size(xcoord)),c,'filled','MarkerEdgeColor','k');
title('move = 0')

for r=1:size(sol,1)
    d=sol(r,2); if d>(4*N)/2, d=d-4*N; end
    for k=1:abs(d)
        idx=rotplane(:,sol(r,1));
        idx1=circshift(idx,sign(d));
        xcoord1=xcoord; xcoord1(idx1)=xcoord1(idx);
        ycoord1=ycoord; ycoord1(idx1)=ycoord1(idx);
        zcoord1=zcoord; zcoord1(idx1)=zcoord1(idx);
        p=12;
        for i=1:p
%             h.XData=xcoord1*i/p+xcoord*(p-i)/p;
%             h.YData=ycoord1*i/p+ycoord*(p-i)/p;
%             h.ZData=zcoord1*i/p+zcoord*(p-i)/p;
            set(h,'XData',xcoord1*i/p+xcoord*(p-i)/p);
            set(h,'YData',ycoord1*i/p+ycoord*(p-i)/p);
            set(h,'ZData',zcoord1*i/p+zcoord*(p-i)/p);
            drawnow % limitrate % display updates
            pause(0.5*1/p)
        end
 
%         % debug
%         h.XData=xcoord1;
%         h.YData=ycoord1;
%         h.ZData=zcoord1;

        pos(idx)=pos(idx1);
        c=zeros(length(xcoord),size(sclr,2));
        for i=1:length(xcoord), c(i,:)=sclr(pos(i),:); end
        h=scatter3(xcoord,ycoord,zcoord,3000*ones(size(xcoord)),c,'filled','MarkerEdgeColor','k');
    end
    title(['move = ' num2str(r)])
end
