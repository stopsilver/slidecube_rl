function [xcoord,ycoord,zcoord,rotplane,cube_rotation_idx]=slidecube_init(N)

%% coordinates for sides
xgrid=linspace(-1,1,N+1);   xgrid=xgrid(1:end-1)+1/N;
ygrid=linspace(-1,1,N+1);   ygrid=ygrid(1:end-1)+1/N;
coordref=zeros(N,N,3);
coordref(:,:,1) = repmat(xgrid,N,1);
coordref(:,:,2) = repmat(ygrid',1,N);
coordref(:,:,3) = 1;

coord=zeros(N,N,3,6);
coord(:,:,:,1)=coordref;                                                   % up
coord(:,:,:,2)=coordref(:,:,[1 3 2]);  coord(:,:,2,2)=-coord(:,:,2,2);     % left
coord(:,:,:,3)=coordref(:,:,[3 1 2]);                                      % front
coord(:,:,:,4)=coordref(:,:,[1 3 2]);                                      % right
coord(:,:,:,5)=coordref(:,:,[3 1 2]);  coord(:,:,1,5)=-coord(:,:,1,5);     % back
coord(:,:,:,6)=coordref;               coord(:,:,3,6)=-coord(:,:,3,6);     % down

if 0
    % debug plot
    clr='bgrcmk';
    figure(1)
    for i=1:6
        plot3(reshape(coord(:,:,1,i),1,[]),reshape(coord(:,:,2,i),1,[]),reshape(coord(:,:,3,i),1,[]),['-o' clr(i)])
        hold on
    end
    hold off
end

coord=permute(coord,[2 1 3 4]);

xcoord=reshape(squeeze(coord(:,:,1,:)),1,[]);
ycoord=reshape(squeeze(coord(:,:,2,:)),1,[]);
zcoord=reshape(squeeze(coord(:,:,3,:)),1,[]);

if 0
    figure(1)
    plot3(xcoord,ycoord,zcoord,'-o')
end

clear coordref coord

%% rotation planes
rotplane=zeros(N*4,N*3);
% x rotation plane
for j=1:N  % all rottion planes in one axis
    idx=find(xcoord==xgrid(j));
    a=ycoord(idx); b=zcoord(idx);
    a=angle(complex(a,b));
    [~,idx1]=sort(a);
    rotplane(:,j+0*N)=idx(idx1)';
end
% y rotation plane
for j=1:N  % all rotation planes in one axis
    idx=find(ycoord==xgrid(j));
    a=xcoord(idx); b=zcoord(idx);
    a=-angle(complex(a,b));
    [~,idx1]=sort(a);
    rotplane(:,j+N)=idx(idx1)';
end
% z rotation plane
for j=1:N  % all rotation planes in one axis
    idx=find(zcoord==xgrid(j));
    a=xcoord(idx); b=ycoord(idx);
    a=angle(complex(a,b));
    [~,idx1]=sort(a);
    rotplane(:,j+2*N)=idx(idx1)';
end

%% check rotation
if 0
    s=kron('bgrcmk',ones(1,4));
    % rotation
    idx=6;   rot=1;
    idx=rotplane(:,idx);
    s1=s;
    s1(idx)=s1(circshift(idx,rot));

    figure(1)
    for i=1:length(s), plot3(xcoord(i),ycoord(i),zcoord(i),['.' s(i)],'MarkerSize',200); hold on; end
    hold off

    figure(2)
    for i=1:length(s1), plot3(xcoord(i),ycoord(i),zcoord(i),['.' s1(i)],'MarkerSize',200); hold on; end
    hold off
end

%% entire cube rotation
cube_rotation_idx=zeros(6*N^2,3);

% around X axis
idx=find(xcoord==-1);  a=ycoord(idx); b=zcoord(idx);a=angle(complex(a,b));
[~,idxm1]=sort(a);   idxm1=idx(idxm1);
idx=find(xcoord==1);  a=ycoord(idx); b=zcoord(idx); a=angle(complex(a,b));
[~,idxp1]=sort(a);   idxp1=idx(idxp1);
a=1:6*N^2;
a(idxm1)=a(circshift(idxm1,1)); a(idxp1)=a(circshift(idxp1,1));     % rotate side squares
for i=1:N                                                           % rotate other squares
    idx=rotplane(:,2*N+i);
    for j=1:N
        a(idx)=a(circshift(idx,1));
    end
end
cube_rotation_idx(:,3)=a;

% around Y axis
idx=find(ycoord==-1);  a=xcoord(idx); b=zcoord(idx);a=angle(complex(a,b));
[~,idxm1]=sort(a);   idxm1=idx(idxm1);
idx=find(ycoord==1);  a=xcoord(idx); b=zcoord(idx); a=angle(complex(a,b));
[~,idxp1]=sort(a);   idxp1=idx(idxp1);
a=1:6*N^2;
a(idxm1)=a(circshift(idxm1,1)); a(idxp1)=a(circshift(idxp1,1));     % rotate side squares
for i=1:N                                                           % rotate other squares
    idx=rotplane(:,1*N+i);
    for j=1:N
        a(idx)=a(circshift(idx,1));
    end
end
cube_rotation_idx(:,2)=a;

% around Z axis
idx=find(zcoord==-1);  a=xcoord(idx); b=ycoord(idx);a=angle(complex(a,b));
[~,idxm1]=sort(a);   idxm1=idx(idxm1);
idx=find(zcoord==1);  a=xcoord(idx); b=ycoord(idx); a=angle(complex(a,b));
[~,idxp1]=sort(a);   idxp1=idx(idxp1);
a=1:6*N^2;
a(idxm1)=a(circshift(idxm1,1)); a(idxp1)=a(circshift(idxp1,1));     % rotate side squares
for i=1:N                                                           % rotate other squares
    idx=rotplane(:,0*N+i);
    for j=1:N
        a(idx)=a(circshift(idx,1));
    end
end
cube_rotation_idx(:,1)=a;
