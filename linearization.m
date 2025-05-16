%% Projet colorimétrie - Léo SEVE, Antoine DUTEYRAT

%% Ordre
% - Linéarisation des images JPG (approx. RAW)
% - Détection des patchs (mire)
% - Balance des blancs, matrice Von Kries

%% Linéarisation des images JPG (approx. RAW)

for idx = 1:12
    % Lecture + conversion en double
    inName  = sprintf('data/%d.jpg', idx);
    imRGB   = im2double(imread(inName));
    
    % Linéarisation
    linRGB  = f(imRGB);
    
    % Sauvegarde en TIFF
    outName = sprintf('data/%d_lin.tiff', idx);
    imwrite(linRGB, outName, 'tiff');
    
    % Affichage en console
    fprintf('Linéarisé %s -> %s\n', inName, outName);

    % Affichage image
    figure(1), subplot(3,4,idx), imshow(linRGB), title(sprintf('%d lin',idx));
    drawnow;
end

%% ===================== FONCTIONS =======================

%% Fonction de linéarisation
% function y = f(x)
%     if x > 0.04045
%         y = ((x + 0.055) / 1.055) ^ 2.4;
%     else
%         y = x / 12.92;
%     end
% end

%% Fonction de linéarisation vectorisée
function y = f(x)
    y = zeros(size(x));
    mask = x > 0.04045;
    
    % Branche pour x > 0.04045
    y(mask) = ((x(mask) + 0.055) / 1.055) .^ 2.4;
    
    % Branche pour x <= 0.04045
    y(~mask) = x(~mask) / 12.92;
end
