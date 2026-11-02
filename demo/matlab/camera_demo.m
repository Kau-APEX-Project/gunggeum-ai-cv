% Select any photo and execute EXP005 using local Python CPU inference.
% No lookup of stored predictions. Run this file from the MATLAB Editor.
root = fileparts(mfilename('fullpath'));
base = fileparts(root);
pythonExe = fullfile(base, '.venv-inference', 'Scripts', 'python.exe');
runner = fullfile(root, 'infer_request.py');
assert(isfile(pythonExe), 'Python environment not found: %s', pythonExe);
[file, folder] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp;*.webp','Image files'}, 'Select a palace photo');
if isequal(file,0), return; end
photo = fullfile(folder,file);
resultsDir = fullfile(root,'results');
if ~isfolder(resultsDir), mkdir(resultsDir); end
requestPath = [tempname(resultsDir) '.json'];
fid = fopen(requestPath,'w','n','UTF-8');
assert(fid~=-1,'Cannot write request');
fprintf(fid,'%s',jsonencode(struct('image',photo)));fclose(fid);
fprintf('Running EXP005 on CPU. Loading model and processing photo...\n');
drawnow;
command = sprintf('"%s" "%s" --request "%s"',pythonExe,runner,requestPath);
[status, message] = system(command);
if status~=0, error('Inference failed:\n%s',message); end
[~,stem] = fileparts(requestPath);
resultPath = fullfile(resultsDir,[stem '.result.json']);
r = jsondecode(fileread(resultPath));
figure('Name','Live EXP005 inference','Color','w','Position',[100 100 1200 560]);
subplot(1,2,1);image(imread(r.preview_path));axis image off;
title(file,'Interpreter','none');
subplot(1,2,2);axis([0 1 0 1]);axis off;
text(.02,.9,'Top-3 predictions','FontSize',20);
for j=1:numel(r.candidates)
 c=r.candidates(j);color=[.3 .3 .3];if j==1,color=[.1 .45 .25];end
 text(.02,.73-(j-1)*.17,sprintf('%d  %s  %.2f%%',j,char(c.name_ko),100*c.score),'FontSize',16,'Color',color,'Interpreter','none');
end
text(.02,.14,'Softmax scores','FontSize',11);
text(.02,.06,sprintf('EXP005 epoch 16 | CPU | %.2f s (including model load)',r.elapsed_seconds),'FontSize',9,'Interpreter','none');
fprintf('Actual inference complete. Result: %s\n',resultPath);
