param(
    [Parameter(Mandatory = $true)]
    [string]$AssemblyPath,

    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory
)

$ErrorActionPreference = "Stop"

$interopRoot = "D:\Program Files\SOLIDWORKS Corp\SOLIDWORKS\api\redist"
$sldworksInterop = Join-Path $interopRoot "SolidWorks.Interop.sldworks.dll"
$swconstInterop = Join-Path $interopRoot "SolidWorks.Interop.swconst.dll"

if (-not (Test-Path -LiteralPath $sldworksInterop)) {
    throw "SolidWorks 2024 interop assembly not found: $sldworksInterop"
}
if (-not (Test-Path -LiteralPath $AssemblyPath)) {
    throw "Assembly not found: $AssemblyPath"
}

Add-Type -Path $sldworksInterop
Add-Type -Path $swconstInterop

# Keep all SolidWorks calls inside compiled C#. PowerShell's dynamic COM binder
# asks this installation for broken type information before invoking a method.
Add-Type -ReferencedAssemblies $sldworksInterop, $swconstInterop -TypeDefinition @'
using System;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using SolidWorks.Interop.sldworks;
using SolidWorks.Interop.swconst;

public static class Uz05SolidWorksExport
{
    private static readonly CultureInfo Invariant = CultureInfo.InvariantCulture;

    public static string Export(object applicationObject, string assemblyPath, string outputDirectory)
    {
        ISldWorks application = (ISldWorks)applicationObject;
        int errors = 0;
        int warnings = 0;
        IModelDoc2 model = application.OpenDoc6(
            Path.GetFullPath(assemblyPath),
            (int)swDocumentTypes_e.swDocASSEMBLY,
            (int)swOpenDocOptions_e.swOpenDocOptions_ReadOnly |
                (int)swOpenDocOptions_e.swOpenDocOptions_Silent,
            "",
            ref errors,
            ref warnings);

        if (model == null)
            throw new InvalidOperationException(
                "SolidWorks could not open the assembly (errors=" + errors +
                ", warnings=" + warnings + "): " + assemblyPath);

        Directory.CreateDirectory(outputDirectory);
        IAssemblyDoc assembly = (IAssemblyDoc)model;
        string configuration = model.ConfigurationManager.ActiveConfiguration.Name;

        WriteDocument(model, configuration, errors, warnings, outputDirectory);
        int componentCount = WriteComponents(assembly, outputDirectory);
        int mateRowCount = WriteMates(model, outputDirectory);
        WriteAssemblyMass(model, outputDirectory);

        return "title=" + model.GetTitle() +
            "; configuration=" + configuration +
            "; components=" + componentCount +
            "; mate_rows=" + mateRowCount +
            "; open_errors=" + errors +
            "; open_warnings=" + warnings;
    }

    private static void WriteDocument(
        IModelDoc2 model,
        string configuration,
        int errors,
        int warnings,
        string outputDirectory)
    {
        using (StreamWriter writer = CsvWriter(Path.Combine(outputDirectory, "document.csv")))
        {
            writer.WriteLine("assembly_path,configuration,open_errors,open_warnings");
            writer.WriteLine(Row(
                model.GetPathName(),
                configuration,
                errors,
                warnings));
        }
    }

    private static int WriteComponents(IAssemblyDoc assembly, string outputDirectory)
    {
        object[] components = assembly.GetComponents(false) as object[];
        int count = 0;
        using (StreamWriter writer = CsvWriter(Path.Combine(outputDirectory, "components.csv")))
        {
            writer.WriteLine("instance,path,configuration,suppression_state,transform_4x4");
            if (components == null)
                return count;

            foreach (object item in components)
            {
                IComponent2 component = item as IComponent2;
                if (component == null)
                    continue;

                double[] transform = null;
                IMathTransform mathTransform = component.Transform2;
                if (mathTransform != null)
                    transform = mathTransform.ArrayData as double[];

                writer.WriteLine(Row(
                    component.Name2,
                    component.GetPathName(),
                    component.ReferencedConfiguration,
                    component.GetSuppression(),
                    ArrayText(transform)));
                count++;
            }
        }
        return count;
    }

    private static int WriteMates(IModelDoc2 model, string outputDirectory)
    {
        int rows = 0;
        using (StreamWriter writer = CsvWriter(Path.Combine(outputDirectory, "mates.csv")))
        {
            writer.WriteLine("mate_name,mate_type,alignment,entity_index,component,entity_params,error_code,suppressed,reference_type,reference_class,face_id,surface_type,surface_params,persist_reference_base64");
            IFeature feature = model.FirstFeature() as IFeature;
            while (feature != null)
            {
                if (feature.GetTypeName2() == "MateGroup")
                {
                    IFeature child = feature.GetFirstSubFeature() as IFeature;
                    while (child != null)
                    {
                        IMate2 mate = null;
                        try { mate = child.GetSpecificFeature2() as IMate2; }
                        catch (COMException) { }

                        if (mate != null)
                        {
                            int entityCount = 0;
                            try { entityCount = mate.GetMateEntityCount(); }
                            catch (COMException) { }

                            if (entityCount == 0)
                            {
                                writer.WriteLine(Row(
                                    child.Name,
                                    mate.Type,
                                    mate.Alignment,
                                    "",
                                    "",
                                    "",
                                    child.GetErrorCode(),
                                    child.IsSuppressed(),
                                    "", "", "", "", "", ""));
                                rows++;
                            }
                            else
                            {
                                for (int index = 0; index < entityCount; index++)
                                {
                                    IMateEntity2 entity = null;
                                    try { entity = mate.MateEntity(index); }
                                    catch (COMException) { }

                                    string componentName = "";
                                    double[] parameters = null;
                                    int referenceType = -1;
                                    object reference = null;
                                    string referenceClass = "";
                                    string faceId = "";
                                    string surfaceType = "";
                                    double[] surfaceParameters = null;
                                    string persistReference = "";
                                    if (entity != null)
                                    {
                                        parameters = entity.EntityParams as double[];
                                        referenceType = entity.ReferenceType2;
                                        reference = entity.Reference;
                                        if (reference != null)
                                        {
                                            referenceClass = reference.GetType().FullName;
                                            IFace2 face = reference as IFace2;
                                            if (face != null)
                                            {
                                                faceId = face.GetFaceId().ToString(Invariant);
                                                ISurface surface = face.GetSurface() as ISurface;
                                                if (surface != null)
                                                {
                                                    if (surface.IsCylinder())
                                                    {
                                                        surfaceType = "cylinder";
                                                        surfaceParameters = surface.CylinderParams as double[];
                                                    }
                                                    else if (surface.IsPlane()) surfaceType = "plane";
                                                    else if (surface.IsCone()) surfaceType = "cone";
                                                    else if (surface.IsSphere()) surfaceType = "sphere";
                                                    else if (surface.IsTorus()) surfaceType = "torus";
                                                    else surfaceType = "other";
                                                }
                                            }
                                            try
                                            {
                                                byte[] persistent = model.Extension.GetPersistReference3(reference) as byte[];
                                                if (persistent != null)
                                                    persistReference = Convert.ToBase64String(persistent);
                                            }
                                            catch (COMException) { }
                                        }
                                        IComponent2 component = entity.ReferenceComponent;
                                        if (component != null)
                                            componentName = component.Name2;
                                    }

                                    writer.WriteLine(Row(
                                        child.Name,
                                        mate.Type,
                                        mate.Alignment,
                                        index,
                                        componentName,
                                        ArrayText(parameters),
                                        child.GetErrorCode(),
                                        child.IsSuppressed(),
                                        referenceType,
                                        referenceClass,
                                        faceId,
                                        surfaceType,
                                        ArrayText(surfaceParameters),
                                        persistReference));
                                    rows++;
                                }
                            }
                        }
                        child = child.GetNextSubFeature() as IFeature;
                    }
                }
                feature = feature.GetNextFeature() as IFeature;
            }
        }
        return rows;
    }

    private static void WriteAssemblyMass(IModelDoc2 model, string outputDirectory)
    {
        IModelDocExtension extension = model.Extension;
        IMassProperty mass = extension.CreateMassProperty();
        mass.UseSystemUnits = true;
        using (StreamWriter writer = CsvWriter(Path.Combine(outputDirectory, "mass_properties.csv")))
        {
            writer.WriteLine("mass_kg,volume_m3,area_m2,center_of_mass_m,inertia_at_com_kg_m2,principal_moments_kg_m2");
            writer.WriteLine(Row(
                mass.Mass,
                mass.Volume,
                mass.SurfaceArea,
                ArrayText(mass.CenterOfMass as double[]),
                ArrayText(mass.GetMomentOfInertia((int)swMassPropertyMoment_e.swMassPropertyMomentAboutCenterOfMass) as double[]),
                ArrayText(mass.PrincipleMomentsOfInertia as double[])));
        }
    }

    private static StreamWriter CsvWriter(string path)
    {
        return new StreamWriter(path, false, new UTF8Encoding(false));
    }

    private static string ArrayText(double[] values)
    {
        if (values == null)
            return "";
        string[] text = new string[values.Length];
        for (int i = 0; i < values.Length; i++)
            text[i] = values[i].ToString("G17", Invariant);
        return string.Join(";", text);
    }

    private static string Row(params object[] values)
    {
        string[] cells = new string[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            string text = Convert.ToString(values[i], Invariant) ?? "";
            cells[i] = "\"" + text.Replace("\"", "\"\"") + "\"";
        }
        return string.Join(",", cells);
    }
}
'@

$application = [Runtime.InteropServices.Marshal]::GetActiveObject("SldWorks.Application")
$summary = [Uz05SolidWorksExport]::Export(
    $application,
    (Resolve-Path -LiteralPath $AssemblyPath).Path,
    [IO.Path]::GetFullPath($OutputDirectory)
)
Write-Output $summary
