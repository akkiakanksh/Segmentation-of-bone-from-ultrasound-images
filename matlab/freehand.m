#used to generate masks(bone area) in a set of images 




close all;  clear all;  clc;
New_Directory = 'C:/Users/akkia/OneDrive/Desktop/test';                                 
folder = dir(New_Directory);
elements = numel(folder) ;
counter = 0;
for k = 3 : elements
   counter = counter+1; 
   filename = strcat(fullfile(New_Directory ,folder(k).name));
   I=imread(filename);  
   GetName{counter,1} = folder(k).name; 
   imshow(I);
   M = imfreehand(gca,'Closed',0);
   F = false(size(M.createMask));
   P0 = M.getPosition;
   D = round([0; cumsum(sum(abs(diff(P0)),2))]); % Need the distance between points...
   P = interp1(D,P0,D(1):.5:D(end)); % ...to close the gaps
   P = unique(round(P),'rows');
   S = sub2ind(size(I),P(:,2),P(:,1));
   F(S) = true;
   figure(1);
   imshow(F);
   imwrite(F,folder(k).name);
end
