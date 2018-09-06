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
## @deftypefn {} {@var{retval} =} dicom2grayMulti (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Erikson <erikson@lagrange>
## Created: 2018-09-05

function [sec1,sec2,sec3,sec4] = dicom2grayMulti (image_dicom)
  
  pkg load dicom;
  pkg load image;
  
  im = dicomread(image_dicom);
  im2 = im2uint8(im);
  imgray = imadjust(im2, stretchlim(im2, 0), []);
  
  %imgray = im2double(im);
  %/home/erikson/Documentos/Dataset/LUNG1-001/09-18-2008-StudyID-69331/0-82046/000001.dcm'
  %disp(X); %print no octave
  %disp(imgray);
  %imshow(imgray, []); %exibe a imagem
  
  %cont = 64;
  %Im = [];
  %Im = [Im, imgray(1:64,1:64)];
  %for i=1:8 
    %cont *= i;
    %Im = cat(64,Im, imgray(1:cont,1:cont));
  %end
  
  row = size(imgray,1);
  col = size(imgray,2);
  
  s = row/4;
  s1 = col/4;
  
  sec1 = imgray(1:s,1:s1);
  sec2 = imgray(s:(2*s),s1:(2*s1));
  sec3 = imgray((2*s):(3*s),(2*s1):(3*s1));
  sec4 = imgray((3*s):(4*s),(3*s1):(4*s1));
  
  
  return ;

endfunction
