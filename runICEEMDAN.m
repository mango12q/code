function [imfs, residue, info] = runICEEMDAN(signal, varargin)
    % RUNICEEMDAN: Optimized ICEEMDAN with Parallel Computing and Mirror Extension
    %
    % Inputs:
    %   signal - Input signal vector
    %   varargin - Name-value pairs for configuration
    %
    % Outputs:
    %   imfs - Intrinsic Mode Functions
    %   residue - Residual signal
    %   info - Decomposition info
    
    signal = signal(:);
    n = numel(signal);
    
    parser = inputParser;
    addParameter(parser, 'NumRealizations', 200, @isnumeric);
    addParameter(parser, 'NoiseStd', 0.2, @isnumeric);
    addParameter(parser, 'MaxIMFs', 10, @isnumeric);
    addParameter(parser, 'ResidualEnergyTol', 1e-5, @isnumeric);
    addParameter(parser, 'Verbose', false, @islogical);
    addParameter(parser, 'ExtensionRatio', 0.0, @isnumeric); 
    addParameter(parser, 'BetaSchedule', [], @(x) true);
    parse(parser, varargin{:});
    opts = parser.Results;
    
    if exist('emd', 'file') ~= 2
        error('Signal Processing Toolbox (emd function) is required.');
    end

    % --- Mirror Extension to reduce edge effects ---
    nExt = floor(n * opts.ExtensionRatio);
    if nExt > 0
        leftExt = flipud(signal(2 : nExt+1));   
        rightExt = flipud(signal(end-nExt : end-1));
        signalProc = [leftExt; signal; rightExt];
        if opts.Verbose
            fprintf('[ICEEMDAN] Mirror Extension (Ratio=%.2f): %d -> %d points\n', ...
                opts.ExtensionRatio, n, numel(signalProc));
        end
    else
        signalProc = signal;
    end
    
    nProc = numel(signalProc);
    
    % --- Step 1: Generate Noise Realizations (Parallelized) ---
    if opts.Verbose, fprintf('    Generating noise realizations...\n'); end
    
    noiseIMFs = cell(opts.NumRealizations, 1);
    maxImfs = opts.MaxIMFs; % Local variable for parfor
    
    % Check if parallel pool exists, if not create one (optional, user might control this)
    % if isempty(gcp('nocreate')), parpool; end 
    
    parfor i = 1:opts.NumRealizations
        omega = randn(nProc, 1);
        noiseIMFs{i} = computeImfs(omega, maxImfs);
    end
    
    imfsProc = zeros(nProc, opts.MaxIMFs);
    residueProc = signalProc;
    
    betaFun = opts.BetaSchedule;
    if isempty(betaFun), betaFun = @(k) opts.NoiseStd / k; end
    
    info = struct('NumIMFs', 0);
    
    % --- Step 2: Iterative Decomposition ---
    for k = 1:opts.MaxIMFs
        beta_k = betaFun(k);
        ensembleIMFs = zeros(nProc, opts.NumRealizations);
        
        if opts.Verbose
            fprintf('    Computing IMF %d ...\n', k);
        end
        
        % Parallelize the ensemble averaging step
        parfor i = 1:opts.NumRealizations
            noiseComp = getImfColumn(noiseIMFs{i}, k);
            noisySig = residueProc + beta_k * noiseComp;
            tmpImf = computeImfs(noisySig, 1); 
            ensembleIMFs(:, i) = tmpImf(:, 1);
        end
        
        currentIMF = mean(ensembleIMFs, 2);
        imfsProc(:, k) = currentIMF;
        residueProc = residueProc - currentIMF;
        info.NumIMFs = info.NumIMFs + 1;
        
        % Check stopping criteria (based on original segment)
        if nExt > 0
            currResSegment = residueProc(nExt+1 : end-nExt);
        else
            currResSegment = residueProc;
        end
        
        if norm(currResSegment) < opts.ResidualEnergyTol || isMonotonic(currResSegment)
            imfsProc = imfsProc(:, 1:k);
            break;
        end
    end
    
    % --- Truncate back to original length ---
    if nExt > 0
        startIdx = numel(leftExt) + 1;
        endIdx = startIdx + n - 1;
        imfs = imfsProc(startIdx:endIdx, :);
        residue = residueProc(startIdx:endIdx);
    else
        imfs = imfsProc;
        residue = residueProc;
    end
end

% --- Local Helper Functions ---

function imfMatrix = computeImfs(signal, maxImfs)
    [c, ~] = emd(signal, 'MaxNumIMF', maxImfs, 'Display', 0);
    if isempty(c)
        imfMatrix = zeros(numel(signal), maxImfs); return;
    end
    cols = min(size(c, 2), maxImfs);
    imfMatrix = zeros(numel(signal), maxImfs);
    imfMatrix(:, 1:cols) = c(:, 1:cols);
end

function imf = getImfColumn(imfMatrix, k)
    if size(imfMatrix, 2) >= k, imf = imfMatrix(:, k);
    else, imf = zeros(size(imfMatrix, 1), 1); end
end

function tf = isMonotonic(x)
    d = diff(x); tf = all(d >= 0) || all(d <= 0);
end
