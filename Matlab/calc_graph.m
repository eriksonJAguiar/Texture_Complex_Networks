## Copyright (C) 2018 Erikson
## 
## This program is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*- 
## @deftypefn {} {@var{retval} =} calc_graph (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Erikson <erikson@lagrange>
## Created: 2018-09-03

%Using package Statistics

% m_width

function [m_adj] = calc_graph (gray_image)    
    row = size(gray_image,1);
    col = size(gray_image,2);
    %[Y, X] = ndgrid(1:row, 1:col);   %indices array
    %x = X(:);
    %y = Y(:);
    %disp(x);
    %disp(y);
    %d = pdist2(gray_image[x,y], gray_image[x+1,y+1], "euclidean");  %pairwise euclidean distances
    %greyval = double(gray_image(:));
    %f = pdist2(greyval, greyval, "euclidean");          %pairwise grayscale distance %FIXED
    %graph = (epsilon1 + f) ./ d;
    %graph(f==0) = epsilon2 ./ d(f==0);  %fix up case of equal intensities
    
    r = 3;
    
    for i = 1:(row - 2)
      for j=1:(col - 2)
        pixel = gray_image(i,j);
        neighboor1 = gray_image(i+1,j);
        neighboor2 = gray_image(i,j+1);
        neighboor3 = gray_image(i+1,j+1);
        d1 = pdist2(pixel, neighboor1);
        d2 = pdist2(pixel,neighboor2);
        d3 = o
      end 
    end
    
    #d = bwdist(gray_image,"euclidean");
    
    #m_adj = d(:,:) <= r;
    
    %m_width = gray_image ./2;
    
   
    
    return
    
endfunction
