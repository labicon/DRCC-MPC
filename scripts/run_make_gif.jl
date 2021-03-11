ENV["PYCALL_JL_RUNTIME_PYTHON"] = Sys.which("python")

using ArgParse
using FileIO
using JLD2
using RiskSensitiveSAC
import RobotOS.Time
using GR
GR.inline("png") # https://github.com/JuliaPlots/Plots.jl/issues/1723

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--file_path"
            help = "absolute path to the .jld2 file"
            required = true
            range_tester = isfile
        "--process_all"
            help = "make gifs for all .jld2 files in the same directory"
            action = :store_true
        "--filter_str"
            help = "if process_all is true, apply only to files with fiter_str"
            default = nothing
        "--dtplot"
            help = "plot delta t [s]"
            arg_type = Float64
            required = true
        "--fps"
            help = "frame per second"
            arg_type = Int64
            required = true
        "--xmin"
            help = "minimum x value [m]. default = 0.0"
            arg_type = Float64
            default = 0.0;
        "--xmax"
            help = "maximum x value [m]. default = 20.0"
            arg_type = Float64
            default = 20.0;
        "--ymin"
            help = "minimum y value [m]. default = 0.0"
            arg_type = Float64
            default = 0.0;
        "--ymax"
            help = "maximum y value [m]. default = 20.0"
            arg_type = Float64
            default = 20.0;
        "--height"
            help = "height of the figure. default = 400"
            arg_type = Int64
            default = 400
        "--width"
            help = "width of the figure. default = 600"
            arg_type = Int64
            default = 600
        "--legend"
            help = "legend position. default = topright"
            arg_type = String
            default = "topright"
        "--legendfontsize"
            help = "legend font size. default = 7"
            arg_type = Int64
            default = 7
        "--markersize"
            help = "markersize. default = 8.0"
            arg_type = Float64
            default = 8.0
        "--show_prediction"
            help = "show predictions."
            action = :store_true
        "--show_nominal_trajectory"
            help = "show nominal ego trajectory."
            action = :store_true
        "--show_past_trajectory"
            help = "show past ego trajectory."
            action = :store_true
        "--dummy_pos"
            help = "coordinate to omit from prediction plots."
            arg_type = Float64

    end
    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()
    file_path = parsed_args["file_path"];
    filter_str = parsed_args["filter_str"];
    if parsed_args["process_all"]
        file_dir = splitdir(file_path)[1];
        files = readdir(file_dir)
        if isnothing(filter_str)
            valid_files = filter(file -> occursin(".jld2", file), files);
        else
            filter_str = parsed_args["filter_str"];
            valid_files = filter((file -> occursin(".jld2", file) && occursin(filter_str, file)), files);
        end
        ii = 1;
        for file in valid_files
            println("Progress: $(ii)/$(length(valid_files))")
            valid_file_path = joinpath(file_dir, file);
            file_content = load(valid_file_path);
            result = file_content["result"];
            gif_file_path = replace(valid_file_path, ".jld2" => ".gif");
            make_gif(result,
                    dtplot=parsed_args["dtplot"],
                    fps=parsed_args["fps"],
                    figsize=(parsed_args["width"], parsed_args["height"]),
                    legendfontsize=parsed_args["legendfontsize"],
                    legend=Symbol(parsed_args["legend"]),
                    xlim=(parsed_args["xmin"], parsed_args["xmax"]),
                    ylim=(parsed_args["ymin"], parsed_args["ymax"]),
                    markersize=parsed_args["markersize"],
                    filename=gif_file_path,
                    show_prediction=parsed_args["show_prediction"],
                    show_nominal_trajectory=parsed_args["show_nominal_trajectory"],
                    show_past_ego_trajectory=parsed_args["show_past_trajectory"],
                    dummy_pos=parsed_args["dummy_pos"]);
            ii += 1;
        end
    else
        file_content = load(file_path);
        result = file_content["result"];
        gif_file_path = replace(file_path, ".jld2" => ".gif");
        make_gif(result,
                dtplot=parsed_args["dtplot"],
                fps=parsed_args["fps"],
                figsize=(parsed_args["width"], parsed_args["height"]),
                legendfontsize=parsed_args["legendfontsize"],
                legend=Symbol(parsed_args["legend"]),
                xlim=(parsed_args["xmin"], parsed_args["xmax"]),
                ylim=(parsed_args["ymin"], parsed_args["ymax"]),
                markersize=parsed_args["markersize"],
                filename=gif_file_path,
                show_prediction=parsed_args["show_prediction"],
                show_nominal_trajectory=parsed_args["show_nominal_trajectory"],
                show_past_ego_trajectory=parsed_args["show_past_trajectory"],
                dummy_pos=parsed_args["dummy_pos"]);
    end
end

main()
