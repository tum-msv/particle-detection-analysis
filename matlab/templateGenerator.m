classdef templateGenerator
    %templateGenerator Generate a template representing all patterns
    % Input:
    % patterns:       matrix which contains extracted patterns
    % opt:            contains all options for processing
    % Internal:
    % patterns:       same as input 'patterns'
    % n_patterns:     number of patterns
    % Np:             number of samples per pattern
    % th:             threshold for witdth split    
    % th_mu:          percentage above/below 
    % f_s:            sampling rate
    % sig:            variance of the Gaussian kernels
    % n_coeff:        number of kernels N'
    % lambda:         regularization parameter
    % tmplt_gauss:    contains the derived kernel regression pattern
    % mse_gauss:      resulting mse of gauss template 
    % tmplt_mean:     mean pattern of all patterns
    % splitopt:       procedure how to split patterns into different classes
    properties
        patterns
        n_patterns
        Np
        th
        th_mu
        f_s
        sig
        n_coeff
        lambda
        coeff_gauss
        tmplt_gauss
        mse_gauss
        tmplt_mean
        splitopt
    end
    
    methods
        function obj = templateGenerator(patterns, opt)
            %constructor Construct an instance of this class
            obj.patterns = patterns;
            [obj.n_patterns, obj.Np] = size(patterns);
            obj.th = opt.th_split;
            obj.th_mu = opt.th_class;
            obj.f_s = opt.f_s;
            obj.sig = opt.sig;
            obj.n_coeff = opt.n_coeff;
            obj.lambda = opt.lambda * numel(obj.patterns);
            obj.splitopt = opt.splitopt;
            classIdx = obj.splitPatterns();
            obj.tmplt_gauss = zeros(obj.Np,max(classIdx));
            for i = 1:max(classIdx)
                [obj.tmplt_gauss(:,i), obj.coeff_gauss] = ...
                   obj.kernelReg(obj.patterns(classIdx==i,:), 'gauss');
            end
            obj.tmplt_mean = mean(obj.patterns, 1);
            disp('Template calculated.');
        end
        
        function classIdx = splitPatterns(obj)
            P = size(obj.patterns,1);
            if strcmp(obj.splitopt,'mean')
                p = obj.patterns ./ sqrt(sum(obj.patterns.^2,2));
                startIdx = zeros(P,1); endIdx = zeros(P,1);
                % find out width of every pattern
                for i = 1:size(p,1)
                    idx = find(p(i,:)<-obj.th, 1, 'last');
                    endIdx(i) = idx + find(p(i,idx:end)>-obj.th/2,1,'first') - 1;
                    idx = find(p(i,:)>obj.th, 1, 'first');
                    startIdx(i) = find(p(i,1:idx)<obj.th/2, 1, 'last');
                end
                width = endIdx - startIdx + 1;
                mu = mean(width);
                % assign to three classes depending on length
                classIdx = zeros(P,1);
                classIdx(width <= (1-obj.th_mu)*mu) = 1;
                classIdx(and(width>(1-obj.th_mu)*mu,width<(1+obj.th_mu)*mu)) = 2;
                classIdx(width >= (1+obj.th_mu)*mu) = 3;
            elseif strcmp(obj.splitopt,'manual')
                % split manually into two classes
                classIdx = zeros(P,1);
                nouseIdx = [46,83,96,103];
                specialIdx = [19,67,104]; %[1,8,19,48,56,66,67,104];
                classIdx(specialIdx) = 1;
                classIdx(classIdx==0) = 2;
                classIdx(nouseIdx) = 0;
            elseif strcmp(obj.splitopt,'everything')
                % use every pattern to create just one template
                classIdx = ones(P,1);
            else
                error('Specify valid pattern splitting procedure!');
            end
        end
        
        function [tmplt, coeff] = kernelReg(obj, p, kerneltype)
            %kernelReg Calculates least squares kernel
            % Input:
            % kerneltype   string   type of kernel, e.g. 'gauss'
            if strcmp(kerneltype, 'gauss')
                % define Gaussian kernel matrix
                K_single = createGaussMatrix(obj.Np, obj.n_coeff, obj.sig);
            else
                error('Choose valid kernel!');
            end
            % define left side of equation (all samples in vector)
            P = size(p,1);
            F = p.'; F = F(:);
            % solve least squares problem
            K = kron(ones(P,1), K_single);
            coeff = (K' * K + obj.lambda*eye(size(K,2))) \ K' * F;
            % calculate kernel/template (as row vector)
            tmplt = K_single * coeff;
            tmplt = tmplt / norm(tmplt);
            % calculate obtained mse
            % err = mean((F - repmat(tmplt(:), obj.n_patterns, 1)).^2);
        end
        
        function plot(obj)
            %plot Creates a plot of all acquired templates
            f_handle = figure;
            a_handle = axes('XScale', 'linear', 'YScale', 'linear');
            hold on, grid on
            t = (0:obj.Np-1) / obj.f_s;
            if ~isempty(obj.tmplt_gauss)
                for i = 1:size(obj.tmplt_gauss,2)
                    plot(t, obj.tmplt_gauss(:,i), 'LineStyle', '-',...
                        'LineWidth', 2, 'DisplayName', ['Gauss ' int2str(i)])
                end
            end
            if ~isempty(obj.tmplt_mean)
                plot(t, obj.tmplt_mean, 'Color', 'r', 'LineStyle', '--',...
                    'LineWidth', 2, 'DisplayName', 'Mean')
            end
            xlabel('t [s]'), ylabel('norm. signal')
            legend('show', 'Location', 'northeast');
            hold off
        end
    end
    
end